"""Torch training function for the credal DRO method."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from probly.method.credal_dro._common import credal_dro_deltas
from probly.train.credal.torch import cvar_ce_loss
from probly.train.torch import train_model

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

    from probly.method.credal_dro._common import CredalDROPredictor
    from probly.train.torch import EpochHook, OptimizerFactory, SchedulerFactory

# The probly_benchmark CIFAR-10 recipe.
_DEFAULT_OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=5e-4)


def train_credal_dro[**In, Out](
    predictor: CredalDROPredictor[In, Out],
    train_loader: DataLoader,
    *,
    val_loader: DataLoader | None = None,
    delta_g: float = 0.5,
    epochs: int = 10,
    optimizer_factory: OptimizerFactory = _DEFAULT_OPTIMIZER_FACTORY,
    scheduler_factory: SchedulerFactory | None = None,
    device: torch.device | str | None = None,
    on_epoch: EpochHook | None = None,
) -> CredalDROPredictor[In, Out]:
    """Train a credal DRO ensemble following Algorithm 1 of :cite:`wangLearningCredalEnsembles2026`.

    Member ``i`` minimizes the CVaR cross-entropy at level ``credal_dro_deltas(delta_g, num_members)[i]``: only
    the worst ``delta_i`` fraction of each batch receives gradient, and the member at level 1 is a plain ERM model.

    Args:
        predictor: The ``credal_dro`` ensemble; members are trained in place.
        train_loader: Loader yielding ``(inputs, targets)`` batches.
        val_loader: Optional validation loader; adds the member's ``"val_loss"`` (same CVaR objective).
        delta_g: Global worst-case CVaR level in (0, 1]. Default is 0.5.
        epochs: Number of epochs per member. Default is 10.
        optimizer_factory: Optimizer factory applied per member. Default follows the probly_benchmark CIFAR-10
            recipe: SGD with learning rate 0.1, momentum 0.9, weight decay 5e-4.
        scheduler_factory: Scheduler factory applied per member, stepped once per epoch.
        device: If given, move each member and every batch to this device.
        on_epoch: Per-epoch hook receiving ``{"member": ..., "delta": ..., "epoch": ..., "running_loss": ...}``;
            returning True stops that member early.

    Returns:
        The trained predictor.
    """
    members = list(predictor)
    deltas = credal_dro_deltas(delta_g, len(members))
    for i, (member, delta) in enumerate(zip(members, deltas, strict=True)):
        train_model(
            cast("nn.Module", member),
            train_loader,
            partial(cvar_ce_loss, delta=delta),
            val_loader=val_loader,
            epochs=epochs,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            device=device,
            on_epoch=on_epoch,
            extra_metrics={"member": float(i), "delta": delta},
        )
    return predictor


__all__ = ["train_credal_dro"]
