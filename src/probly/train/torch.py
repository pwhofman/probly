"""Generic torch training blocks shared by the method training functions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

type EpochHook = Callable[[dict[str, float]], bool | None]
"""Per-epoch callback: receives the epoch's metrics; returning True stops training."""

type OptimizerFactory = Callable[..., Optimizer]
"""Builds an optimizer from an iterable of parameters, e.g. ``partial(SGD, lr=0.1)``."""

type SchedulerFactory = Callable[[Optimizer], LRScheduler]
"""Builds a scheduler from an optimizer, e.g. ``partial(CosineAnnealingLR, T_max=50)``."""

_DEFAULT_OPTIMIZER_FACTORY: OptimizerFactory = partial(torch.optim.Adam, lr=1e-3)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device | str | None,
) -> float:
    """Sample-weighted mean ``loss_fn`` of the model over the loader, without gradients."""
    was_training = model.training
    model.eval()
    total = 0.0
    num_samples = 0
    for batch_inputs, batch_targets in loader:
        inputs = batch_inputs.to(device)
        targets = batch_targets.to(device)
        total += loss_fn(model(inputs), targets).item() * inputs.shape[0]
        num_samples += inputs.shape[0]
    model.train(was_training)
    return total / max(num_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    val_loader: DataLoader | None = None,
    epochs: int = 10,
    optimizer_factory: OptimizerFactory = _DEFAULT_OPTIMIZER_FACTORY,
    scheduler_factory: SchedulerFactory | None = None,
    device: torch.device | str | None = None,
    on_epoch: EpochHook | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> nn.Module:
    """Minimize ``loss_fn(model(inputs), targets)`` over the loader for at most ``epochs`` epochs.

    The minimal supervised loop underlying the ``train_*`` method trainers. After every epoch, ``on_epoch``
    receives ``{**extra_metrics, "epoch": ..., "running_loss": ...}``, plus ``"val_loss"`` when a validation
    loader is given; returning True stops training early. Logging, progress bars, and early stopping are all
    expressed through this single hook (e.g. ``on_epoch=wandb.log``).

    Args:
        model: The model to train in place.
        train_loader: Loader yielding ``(inputs, targets)`` batches.
        loss_fn: Per-batch loss on ``(output, targets)``.
        val_loader: Optional validation loader; adds ``"val_loss"``, the mean ``loss_fn`` over it, to the metrics.
        epochs: Maximum number of epochs. Default is 10.
        optimizer_factory: Callable mapping ``model.parameters()`` to an optimizer. Default is Adam with
            learning rate 1e-3.
        scheduler_factory: Callable mapping the optimizer to a scheduler, stepped once per epoch. Default is
            no scheduler.
        device: If given, move the model and every batch to this device.
        on_epoch: Optional per-epoch hook; returning True stops training.
        extra_metrics: Constant entries merged into every metrics dict, e.g. the ensemble member index.

    Returns:
        The trained model, set to eval mode.
    """
    model = model.to(device)
    optimizer = optimizer_factory(model.parameters())
    scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0
        for batch_inputs, batch_targets in train_loader:
            inputs = batch_inputs.to(device)
            targets = batch_targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.shape[0]
            num_samples += inputs.shape[0]
        if scheduler is not None:
            scheduler.step()
        metrics = {
            **(extra_metrics or {}),
            "epoch": float(epoch),
            "running_loss": running_loss / max(num_samples, 1),
        }
        if val_loader is not None:
            metrics["val_loss"] = _evaluate(model, val_loader, loss_fn, device)
        if on_epoch is not None and on_epoch(metrics):
            break
    model.eval()
    return model


__all__ = ["EpochHook", "OptimizerFactory", "SchedulerFactory", "train_model"]
