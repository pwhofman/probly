"""Acquisition functions for Bayesian optimization.

Acquisition optimization uses multi-start L-BFGS-B with analytic gradients
obtained via PyTorch autograd through the surrogate. The starts are taken
from the lowest-scoring entries of a Sobol candidate sweep so that the
optimizer is initialized in promising regions before refining locally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from botorch.utils.sampling import draw_sobol_samples
import numpy as np
from scipy.optimize import minimize as scipy_minimize
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from probly.evaluation.bayesian_optimization.surrogate import Surrogate


@runtime_checkable
class Acquisition(Protocol):
    """Protocol for acquisition strategies.

    An acquisition function consumes a fitted surrogate and the feasible
    bounds and returns the next input to evaluate.
    """

    def select(self, surrogate: Surrogate, bounds: torch.Tensor) -> torch.Tensor:
        """Pick the next input to evaluate.

        Args:
            surrogate: Fitted surrogate exposing ``posterior_mean_std``.
            bounds: Feasible region as a ``(2, dim)`` tensor.

        Returns:
            A single candidate of shape ``(1, dim)``.
        """
        ...


def _sobol_candidates(bounds: torch.Tensor, n: int, seed: int | None) -> torch.Tensor:
    """Return ``n`` Sobol-sampled candidates inside ``bounds``."""
    raw = draw_sobol_samples(bounds=bounds.to(torch.float64), n=n, q=1, seed=seed)
    return raw.squeeze(-2)


def _optimize_acqf(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    *,
    n_restarts: int = 10,
    n_raw_samples: int = 1024,
    seed: int | None = None,
) -> torch.Tensor:
    """Minimize an acquisition score over a box using L-BFGS-B with restarts.

    Strategy:

    1. Sample ``n_raw_samples`` Sobol candidates inside ``bounds`` and
       score them in a single batched forward pass.
    2. Use the ``n_restarts`` lowest-scoring candidates as starting points
       for local L-BFGS-B optimization. Gradients are obtained from the
       PyTorch autograd graph through ``score_fn``.
    3. Return the single best optimized point as a ``(1, dim)`` tensor.

    Args:
        score_fn: Callable mapping ``(n, dim)`` inputs to ``(n,)`` scores.
            Lower scores are better. Must be differentiable through autograd.
        bounds: ``(2, dim)`` tensor with row 0 = lower, row 1 = upper bounds.
        n_restarts: Number of L-BFGS-B restarts.
        n_raw_samples: Sobol sweep size for selecting starting points.
        seed: Optional seed for the Sobol sweep.

    Returns:
        A ``(1, dim)`` tensor giving the best minimizer found.
    """
    bounds64 = bounds.to(torch.float64)
    lo = bounds64[0].detach().cpu().numpy().astype(np.float64)
    hi = bounds64[1].detach().cpu().numpy().astype(np.float64)
    bounds_list = list(zip(lo.tolist(), hi.tolist(), strict=True))

    raw = _sobol_candidates(bounds64, n=n_raw_samples, seed=seed)
    with torch.no_grad():
        raw_scores = score_fn(raw)
    n_starts = min(n_restarts, n_raw_samples)
    top_idx = torch.topk(raw_scores, n_starts, largest=False).indices
    starts = raw[top_idx].detach().cpu().numpy().astype(np.float64)

    def f_and_grad(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        x_t = torch.from_numpy(x_np).to(torch.float64).requires_grad_(True)
        score = score_fn(x_t.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(score, x_t)[0]
        return float(score.detach().item()), grad.detach().cpu().numpy().astype(np.float64)

    best_x: np.ndarray | None = None
    best_v: float = float("inf")
    for start in starts:
        try:
            res = scipy_minimize(f_and_grad, start, method="L-BFGS-B", jac=True, bounds=bounds_list)
        except (RuntimeError, ValueError):
            continue
        if not np.isfinite(res.fun):
            continue
        if float(res.fun) < best_v:
            best_v = float(res.fun)
            best_x = res.x

    if best_x is None:
        best_idx = int(torch.argmin(raw_scores).item())
        return raw[best_idx : best_idx + 1].detach()

    out = torch.from_numpy(best_x).to(torch.float64).unsqueeze(0)
    return torch.minimum(torch.maximum(out, bounds64[0]), bounds64[1])


class RandomAcquisition:
    """Sample the next input uniformly at random from the feasible region.

    Sobol sampling is used so successive seeds produce well-spread points
    without long-range correlations. Useful as a sanity baseline.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize with an optional base seed.

        Args:
            seed: Base seed; each call uses ``seed + step`` for reproducibility.
        """
        self._seed = seed
        self._step = 0

    def select(self, surrogate: Surrogate, bounds: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Return one Sobol-sampled point inside ``bounds``."""
        seed = None if self._seed is None else self._seed + self._step
        self._step += 1
        return _sobol_candidates(bounds, n=1, seed=seed)


class UpperConfidenceBound:
    """UCB acquisition for **minimization**: pick ``argmin(mean - beta * std)``.

    The acquisition is optimized with multi-restart L-BFGS-B using analytic
    gradients flowing through the surrogate's ``posterior_mean_std``. Larger
    ``beta`` favors exploration of high-uncertainty regions; smaller ``beta``
    favors exploitation of regions with low predicted mean.

    Attributes:
        beta: Exploration trade-off coefficient.
        n_restarts: Number of L-BFGS-B restarts per query.
        n_raw_samples: Sobol sweep size used to seed the restarts.
    """

    def __init__(
        self,
        beta: float = 2.0,
        n_restarts: int = 10,
        n_raw_samples: int = 1024,
        seed: int | None = None,
    ) -> None:
        """Construct a UCB acquisition.

        Args:
            beta: Exploration trade-off coefficient.
            n_restarts: Number of L-BFGS-B restarts per query.
            n_raw_samples: Sobol sweep size used to seed the restarts.
            seed: Base seed for the Sobol engine; each call uses ``seed + step``.
        """
        self.beta = beta
        self.n_restarts = n_restarts
        self.n_raw_samples = n_raw_samples
        self._seed = seed
        self._step = 0

    def select(self, surrogate: Surrogate, bounds: torch.Tensor) -> torch.Tensor:
        """Return the minimizer of ``mean - beta * std`` over ``bounds``."""
        seed = None if self._seed is None else self._seed + self._step
        self._step += 1
        beta = self.beta

        def score(x: torch.Tensor) -> torch.Tensor:
            mean, std = surrogate.posterior_mean_std(x)
            return mean - beta * std

        return _optimize_acqf(
            score,
            bounds,
            n_restarts=self.n_restarts,
            n_raw_samples=self.n_raw_samples,
            seed=seed,
        )
