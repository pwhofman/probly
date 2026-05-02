"""Surrogate models for Bayesian optimization.

Defines a backend-agnostic :class:`Surrogate` protocol used by acquisition
functions, plus two implementations:

* :class:`BotorchGPSurrogate` -- exact Gaussian process via botorch
  ``SingleTaskGP`` with input normalization and outcome standardization.
* :class:`EnsembleSurrogate` -- deep ensemble of MLP regressors. Predictive
  mean and standard deviation are computed from the per-member predictions
  and serve as a non-Gaussian alternative to the GP posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class Surrogate(Protocol):
    """Protocol for BO surrogate models.

    A surrogate maintains a probabilistic regression model fit to observed
    ``(x, y)`` pairs. Acquisition functions query :meth:`posterior_mean_std`
    to score candidate inputs.
    """

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the surrogate on observed inputs and outputs.

        Args:
            x: Observed inputs of shape ``(n, dim)``.
            y: Observed outputs of shape ``(n,)`` or ``(n, 1)``.
        """
        ...

    def posterior_mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predictive mean and standard deviation at ``x``.

        Args:
            x: Query inputs of shape ``(n, dim)``.

        Returns:
            Pair ``(mean, std)`` of tensors of shape ``(n,)`` each.
        """
        ...


class BotorchGPSurrogate:
    """Exact GP surrogate using botorch ``SingleTaskGP``.

    Wraps the GP with a :class:`~botorch.models.transforms.input.Normalize`
    input transform and a :class:`~botorch.models.transforms.outcome.Standardize`
    outcome transform so the GP fits roughly unit-scale data regardless of
    the objective.
    """

    def __init__(self) -> None:
        """Construct an unfitted GP surrogate."""
        self._model: SingleTaskGP | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Refit the GP on the full set of observations.

        Args:
            x: Observed inputs of shape ``(n, dim)``.
            y: Observed outputs of shape ``(n,)`` or ``(n, 1)``.
        """
        x_d = x.to(torch.float64)
        y_d = y.to(torch.float64).view(-1, 1)
        d = x_d.shape[-1]
        self._model = SingleTaskGP(
            train_X=x_d,
            train_Y=y_d,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        fit_gpytorch_mll(mll)
        self._model.eval()

    def posterior_mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return GP posterior mean and standard deviation at ``x``.

        Gradients flow through the result so the caller can run an autograd-
        based acquisition optimizer (e.g. L-BFGS-B with analytic gradients).
        Wrap the call in :func:`torch.no_grad` if gradients are not needed.
        """
        if self._model is None:
            msg = "BotorchGPSurrogate must be fit before calling posterior_mean_std."
            raise RuntimeError(msg)
        x_d = x.to(torch.float64)
        posterior = self._model.posterior(x_d)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
        return mean.to(x.dtype), std.to(x.dtype)


class _MLP(nn.Module):
    """A small feedforward regressor used as an ensemble member."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int]) -> None:
        """Build an MLP with the given hidden widths and a scalar output."""
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar prediction at ``x``."""
        return self.net(x).squeeze(-1)


@dataclass
class _EnsembleState:
    """Fitted state of an :class:`EnsembleSurrogate`."""

    members: list[_MLP]
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


class EnsembleSurrogate:
    """Deep ensemble of MLP regressors trained from independent inits.

    Each :meth:`fit` call retrains all members from scratch on the current
    pool of observations -- standard practice for BO surrogates. The
    posterior mean is the cross-member average and the standard deviation
    is the per-input spread across members; the latter is the epistemic
    uncertainty estimate consumed by UCB.

    Attributes:
        num_members: Number of ensemble members.
        hidden_dims: Hidden layer widths for each member.
        lr: Adam learning rate.
        epochs: Full-batch gradient steps per member per fit.
        weight_decay: L2 regularization strength.
        seed: Base RNG seed; the ``k``-th member uses ``seed + k``.
    """

    def __init__(
        self,
        num_members: int = 5,
        hidden_dims: Sequence[int] = (64, 64),
        lr: float = 1e-2,
        epochs: int = 200,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        """Construct an unfitted ensemble surrogate.

        Args:
            num_members: Number of ensemble members.
            hidden_dims: Hidden layer widths for each MLP member.
            lr: Adam learning rate.
            epochs: Full-batch gradient steps per member per fit.
            weight_decay: L2 regularization strength.
            seed: Base RNG seed; the ``k``-th member uses ``seed + k``.
        """
        self.num_members = num_members
        self.hidden_dims = tuple(hidden_dims)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self._state: _EnsembleState | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Retrain all ensemble members on the current observations.

        Args:
            x: Observed inputs of shape ``(n, dim)``.
            y: Observed outputs of shape ``(n,)`` or ``(n, 1)``.
        """
        x_f = x.to(torch.float32)
        y_f = y.to(torch.float32).view(-1)

        x_mean = x_f.mean(dim=0)
        x_std = x_f.std(dim=0).clamp_min(1e-6)
        y_mean = y_f.mean()
        y_std = y_f.std().clamp_min(1e-6)
        x_n = (x_f - x_mean) / x_std
        y_n = (y_f - y_mean) / y_std

        members: list[_MLP] = []
        for k in range(self.num_members):
            torch.manual_seed(self.seed + k)
            member = _MLP(in_dim=x_f.shape[-1], hidden_dims=self.hidden_dims)
            optim = torch.optim.Adam(member.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for _ in range(self.epochs):
                optim.zero_grad()
                pred = member(x_n)
                loss = nn.functional.mse_loss(pred, y_n)
                loss.backward()
                optim.step()
            member.eval()
            members.append(member)

        self._state = _EnsembleState(
            members=members,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )

    def posterior_mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ensemble mean and per-input standard deviation at ``x``.

        Args:
            x: Query inputs of shape ``(n, dim)``.

        Returns:
            Pair ``(mean, std)`` of tensors of shape ``(n,)`` each. The
            standard deviation across members serves as the surrogate's
            epistemic uncertainty estimate.
        """
        if self._state is None:
            msg = "EnsembleSurrogate must be fit before calling posterior_mean_std."
            raise RuntimeError(msg)
        s = self._state
        x_n = (x.to(torch.float32) - s.x_mean) / s.x_std
        preds = torch.stack([m(x_n) for m in s.members], dim=0)
        mean = preds.mean(dim=0) * s.y_std + s.y_mean
        std = preds.std(dim=0) * s.y_std
        return mean.to(x.dtype), std.to(x.dtype)
