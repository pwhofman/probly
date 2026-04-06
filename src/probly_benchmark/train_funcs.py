"""Training functionality for probly benchmark methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast

from lazy_dispatch import lazydispatch
from probly.method.bayesian import BayesianPredictor

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from probly.predictor import Predictor


@lazydispatch
def train_epoch(
    model: Predictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    **kwargs: Any,  # noqa: ANN401
) -> torch.Tensor | float:
    """Train for one epoch."""
    msg = f"No training function for {type(model)}"
    raise NotImplementedError(msg)


@train_epoch.register(BayesianPredictor)
def _(
    model: BayesianPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    scaler: GradScaler | None = None,
) -> torch.Tensor | float:
    """Train a Bayesian predictor for one epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    with autocast(inputs.device.type, enabled=amp_enabled):
        outputs = model(inputs)  # ty: ignore
        loss = criterion(outputs, targets)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)  # ty: ignore[unresolved-attribute]
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)  # ty: ignore[unresolved-attribute]
        optimizer.step()
    return loss.item()


@lazydispatch
def validate(
    model: Predictor,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> float:
    """Validate a model."""
    msg = f"No validation function for {type(model)}"
    raise NotImplementedError(msg)


@validate.register(BayesianPredictor)
@torch.no_grad()
def _(
    model: BayesianPredictor,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> float:
    """Validate a Bayesian predictor."""
    criterion = nn.CrossEntropyLoss()
    model.eval()  # ty: ignore[unresolved-attribute]
    val_loss = 0.0
    for inputs_, targets_ in val_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            outputs = model(inputs)  # ty: ignore[call-non-callable]
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_loader)
    return val_loss


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
