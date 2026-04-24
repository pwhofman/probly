"""Sklearn-compatible wrappers around PyTorch nn.Module for active learning.

Provides four estimator classes:

- :class:`TorchEstimator`: single deterministic model.
- :class:`TorchEnsembleEstimator`: deep ensemble (probly ``ensemble()``).
- :class:`MCDropoutEstimator`: MC-Dropout model (probly ``dropout()``).
- :class:`ProblyEstimator`: credal ensemble with a probly representer and quantifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _default_reset(model: nn.Module) -> None:
    """Reset all layers that expose ``reset_parameters``."""
    for m in model.modules():
        reset = getattr(m, "reset_parameters", None)
        if callable(reset):
            reset()


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Return an explicit torch device, auto-detecting when *device* is None."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchEstimator:
    """Sklearn-style wrapper that adapts a ``torch.nn.Module`` for use with active learning.

    Wraps a :class:`torch.nn.Module` so it exposes the ``fit`` / ``predict`` /
    ``predict_proba`` interface expected by
    :func:`~probly.evaluation.active_learning.active_learning_loop`.

    Args:
        model: A PyTorch module to wrap.
        task: Whether this is a classification or regression model.
        optimizer_cls: Optimizer class used for training.
        optimizer_kwargs: Extra keyword arguments forwarded to the optimizer
            (besides ``params``).
        loss_fn: Loss function. Defaults to ``CrossEntropyLoss`` for
            classification and ``MSELoss`` for regression.
        n_epochs: Number of training epochs per ``fit`` call.
        batch_size: Mini-batch size used during training.
        pred_batch_size: Batch size used during inference.
        device: Device to place the model on. Auto-detected when ``None``.
        reset_fn: Controls weight re-initialisation before each ``fit`` call.
            ``"default"`` resets every layer that has ``reset_parameters``.
            ``None`` keeps the current weights (warm-start).
            A callable receives the model and should reset it in-place.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        task: Literal["classification", "regression"] = "classification",
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        loss_fn: Callable | None = None,
        n_epochs: int = 10,
        batch_size: int = 64,
        pred_batch_size: int = 512,
        device: torch.device | str | None = None,
        reset_fn: Callable[[nn.Module], None] | Literal["default"] | None = "default",
    ) -> None:
        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam

        self.model = model
        self.task = task
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.device = _resolve_device(device)

        if reset_fn == "default":
            self.reset_fn: Callable[[nn.Module], None] | None = _default_reset
        else:
            self.reset_fn = reset_fn

        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        self.model.to(self.device)

        if task == "classification":
            self.predict_proba = self._predict_proba  # type: ignore[assignment]

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """Train the wrapped model on the provided data.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.
            y: Target tensor of shape ``(n_samples,)``.

        Returns:
            ``self`` for method chaining.
        """
        if self.reset_fn is not None:
            self.reset_fn(self.model)

        self.model.train()
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

        x_t = x.to(dtype=torch.float32, device=self.device)
        y_dtype = torch.long if self.task == "classification" else torch.float32
        y_t = y.to(dtype=y_dtype, device=self.device)

        n = len(x_t)
        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                out = self.model(x_t[idx])
                if self.task == "regression":
                    out = out.squeeze(-1)
                loss = self.loss_fn(out, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def _predict_batched(
        self,
        x: torch.Tensor,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run batched inference, apply *transform_fn* per batch, return tensor."""
        self.model.eval()
        x_t = x.to(dtype=torch.float32, device=self.device)
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(x_t), self.pred_batch_size):
                batch = x_t[start : start + self.pred_batch_size]
                out = transform_fn(self.model(batch))
                parts.append(out.cpu())
        return torch.cat(parts, dim=0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions as a tensor.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.

        Returns:
            For classification: integer class labels.
            For regression: float predictions.
        """
        if self.task == "classification":
            return self._predict_batched(x, lambda out: out.argmax(dim=-1))
        return self._predict_batched(x, lambda out: out.squeeze(-1))

    def _predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape ``(n_samples, n_classes)``.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.

        Returns:
            Probability tensor.
        """
        return self._predict_batched(x, lambda out: torch.softmax(out, dim=-1))


# ---------------------------------------------------------------------------
# Deep-ensemble estimator (probly ``ensemble()``)
# ---------------------------------------------------------------------------


def _train_one_epoch(
    model: nn.Module,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    batch_size: int,
    task: str,
    device: torch.device,
) -> None:
    """Run one epoch of mini-batch SGD on a single module in-place."""
    n = len(x_t)
    perm = torch.randperm(n, device=device)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        out = model(x_t[idx])
        if task == "regression":
            out = out.squeeze(-1)
        loss = loss_fn(out, y_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TorchEnsembleEstimator:
    """Sklearn-style wrapper for a deep ensemble of ``torch.nn.Module`` models.

    Uses probly's :func:`~probly.method.ensemble.ensemble` function to create
    *num_members* independently initialised copies of *model*.  Each member is
    trained separately on the full labeled set.  Predictions are obtained by
    averaging the softmax probabilities of all members (classification) or
    averaging the raw outputs (regression).

    Args:
        model: Base PyTorch module.  A deep copy with freshly reset parameters
            is created for each ensemble member.
        num_members: Number of ensemble members.
        task: Whether this is a classification or regression model.
        optimizer_cls: Optimizer class.  Defaults to ``Adam``.
        optimizer_kwargs: Extra keyword arguments forwarded to each member's
            optimizer.
        loss_fn: Loss function.  Defaults to ``CrossEntropyLoss`` /
            ``MSELoss``.
        n_epochs: Training epochs per ``fit`` call per member.
        batch_size: Mini-batch size during training.
        pred_batch_size: Batch size during inference.
        device: Device for all members.  Auto-detected when ``None``.
        reset_fn: Weight reset strategy applied to each member before training.
            ``"default"`` calls ``reset_parameters`` on every layer that has
            it.  ``None`` keeps the current weights (warm-start).  A callable
            receives the member module and resets it in-place.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        num_members: int = 5,
        task: Literal["classification", "regression"] = "classification",
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        loss_fn: Callable | None = None,
        n_epochs: int = 10,
        batch_size: int = 64,
        pred_batch_size: int = 512,
        device: torch.device | str | None = None,
        reset_fn: Callable[[nn.Module], None] | Literal["default"] | None = "default",
    ) -> None:
        from probly.method.ensemble import ensemble  # noqa: PLC0415

        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam

        self.task = task
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.device = _resolve_device(device)

        if reset_fn == "default":
            self.reset_fn: Callable[[nn.Module], None] | None = _default_reset
        else:
            self.reset_fn = reset_fn

        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        # Build ensemble: list of independently initialised module copies.
        self.members: Iterable[nn.Module] = ensemble(model, num_members=num_members)
        for m in self.members:
            m.to(self.device)

        if task == "classification":
            self.predict_proba = self._predict_proba  # type: ignore[assignment]

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """Train each ensemble member independently on the full labeled set.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.
            y: Target tensor of shape ``(n_samples,)``.

        Returns:
            ``self``
        """
        x_t = x.to(dtype=torch.float32, device=self.device)
        y_dtype = torch.long if self.task == "classification" else torch.float32
        y_t = y.to(dtype=y_dtype, device=self.device)

        for member in self.members:
            if self.reset_fn is not None:
                self.reset_fn(member)
            member.train()
            opt = self.optimizer_cls(member.parameters(), **self.optimizer_kwargs)
            for _ in range(self.n_epochs):
                _train_one_epoch(member, x_t, y_t, opt, self.loss_fn, self.batch_size, self.task, self.device)

        return self

    def _predict_proba_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged softmax probabilities, shape ``(n_samples, n_classes)``."""
        x_t = x.to(dtype=torch.float32, device=self.device)
        member_probs = []
        for member in self.members:
            member.eval()
            parts: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(x_t), self.pred_batch_size):
                    batch = x_t[start : start + self.pred_batch_size]
                    out = torch.softmax(member(batch), dim=-1)
                    parts.append(out.cpu())
            member_probs.append(torch.cat(parts, dim=0))
        # Average across members: (K, n, n_classes) -> (n, n_classes)
        return torch.stack(member_probs, dim=0).mean(dim=0)

    def _predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged softmax probabilities of shape ``(n_samples, n_classes)``.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.

        Returns:
            Averaged class probability tensor.
        """
        return self._predict_proba_raw(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions as a tensor.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.

        Returns:
            For classification: integer class labels.
            For regression: mean predictions across ensemble members.
        """
        if self.task == "classification":
            probs = self._predict_proba_raw(x)
            return probs.argmax(dim=-1)

        # Regression: average raw outputs across members.
        x_t = x.to(dtype=torch.float32, device=self.device)
        member_preds = []
        for member in self.members:
            member.eval()
            parts: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(x_t), self.pred_batch_size):
                    batch = x_t[start : start + self.pred_batch_size]
                    out = member(batch).squeeze(-1)
                    parts.append(out.cpu())
            member_preds.append(torch.cat(parts, dim=0))
        return torch.stack(member_preds, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# MC-Dropout estimator (probly ``dropout()``)
# ---------------------------------------------------------------------------


class MCDropoutEstimator:
    """Sklearn-style wrapper for MC-Dropout inference on a ``torch.nn.Module``.

    Uses probly's :func:`~probly.method.dropout.dropout` to insert dropout
    layers into *model*, then keeps the model in *train mode* at inference time
    so that dropout remains active.  Multiple stochastic forward passes are
    averaged to produce a probability estimate.

    Args:
        model: Base PyTorch module.  Dropout layers are inserted before every
            eligible layer via probly's traversal utilities.
        p: Dropout probability applied to each eligible layer.
        num_samples: Number of stochastic forward passes used when computing
            probabilities.
        task: Whether this is a classification or regression model.
        optimizer_cls: Optimizer class.  Defaults to ``Adam``.
        optimizer_kwargs: Extra keyword arguments forwarded to the optimizer.
        loss_fn: Loss function.  Defaults to ``CrossEntropyLoss`` /
            ``MSELoss``.
        n_epochs: Training epochs per ``fit`` call.
        batch_size: Mini-batch size during training.
        pred_batch_size: Batch size per stochastic forward pass.
        device: Device for the model.  Auto-detected when ``None``.
        reset_fn: Weight reset strategy.  ``"default"`` calls
            ``reset_parameters`` on all eligible layers.  ``None`` warm-starts.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        p: float = 0.25,
        num_samples: int = 10,
        task: Literal["classification", "regression"] = "classification",
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        loss_fn: Callable | None = None,
        n_epochs: int = 10,
        batch_size: int = 64,
        pred_batch_size: int = 512,
        device: torch.device | str | None = None,
        reset_fn: Callable[[nn.Module], None] | Literal["default"] | None = "default",
    ) -> None:
        from probly.method.dropout import dropout  # noqa: PLC0415

        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam

        self.task = task
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.num_samples = num_samples
        self.device = _resolve_device(device)

        if reset_fn == "default":
            self.reset_fn: Callable[[nn.Module], None] | None = _default_reset
        else:
            self.reset_fn = reset_fn

        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        self.model: nn.Module = dropout(model, p=p)
        self.model.to(self.device)

        if task == "classification":
            self.predict_proba = self._predict_proba  # type: ignore[assignment]

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Self:
        """Train the dropout model on the provided data.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.
            y: Target tensor of shape ``(n_samples,)``.

        Returns:
            ``self``
        """
        if self.reset_fn is not None:
            self.reset_fn(self.model)

        self.model.train()
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

        x_t = x.to(dtype=torch.float32, device=self.device)
        y_dtype = torch.long if self.task == "classification" else torch.float32
        y_t = y.to(dtype=y_dtype, device=self.device)

        for _ in range(self.n_epochs):
            _train_one_epoch(self.model, x_t, y_t, optimizer, self.loss_fn, self.batch_size, self.task, self.device)

        return self

    def _stochastic_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return average softmax over ``num_samples`` stochastic passes.

        The model is kept in *train* mode so that dropout remains active.

        Returns:
            Tensor of shape ``(n_samples, n_classes)``.
        """
        # Keep train mode to retain dropout stochasticity.
        self.model.train()
        x_t = x.to(dtype=torch.float32, device=self.device)
        sample_probs: list[torch.Tensor] = []
        for _ in range(self.num_samples):
            parts: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(x_t), self.pred_batch_size):
                    batch = x_t[start : start + self.pred_batch_size]
                    out = torch.softmax(self.model(batch), dim=-1)
                    parts.append(out.cpu())
            sample_probs.append(torch.cat(parts, dim=0))
        # shape: (num_samples, n, n_classes) -> average -> (n, n_classes)
        return torch.stack(sample_probs, dim=0).mean(dim=0)

    def _predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return MC-averaged softmax probabilities of shape ``(n_samples, n_classes)``.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.

        Returns:
            Averaged class probability tensor.
        """
        return self._stochastic_probs(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions as a tensor.

        Args:
            x: Feature tensor of shape ``(n_samples, n_features)``.

        Returns:
            For classification: integer class labels.
            For regression: mean predictions across stochastic passes.
        """
        if self.task == "classification":
            return self._stochastic_probs(x).argmax(dim=-1)

        self.model.train()
        x_t = x.to(dtype=torch.float32, device=self.device)
        sample_preds: list[torch.Tensor] = []
        for _ in range(self.num_samples):
            parts: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, len(x_t), self.pred_batch_size):
                    batch = x_t[start : start + self.pred_batch_size]
                    out = self.model(batch).squeeze(-1)
                    parts.append(out.cpu())
            sample_preds.append(torch.cat(parts, dim=0))
        return torch.stack(sample_preds, dim=0).mean(dim=0)
