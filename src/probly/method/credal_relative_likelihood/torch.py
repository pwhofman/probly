"""Torch training function for the credal relative likelihood method."""

from __future__ import annotations

from functools import partial
import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from probly.method.credal_relative_likelihood._common import (
    relative_likelihood_thresholds,
    train_credal_relative_likelihood,
)
from probly.train.torch import train_model

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from probly.train.torch import EpochHook, OptimizerFactory, SchedulerFactory

# The probly_benchmark CIFAR-10 recipe.
_DEFAULT_OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=5e-4)


@torch.no_grad()
def _mean_log_likelihood(model: nn.Module, loader: DataLoader, device: torch.device | str | None) -> float:
    """Mean per-sample log-likelihood of the model on the loader."""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for batch_inputs, batch_targets in loader:
        inputs = batch_inputs.to(device)
        targets = batch_targets.to(device)
        log_probs = F.log_softmax(model(inputs), dim=-1)  # (B, C)
        total += log_probs.gather(-1, targets.unsqueeze(-1)).sum().item()
        count += targets.shape[0]
    model.train(was_training)
    return total / max(count, 1)


@train_credal_relative_likelihood.register((nn.ModuleList, list))
def train_credal_relative_likelihood_torch(
    predictor: nn.ModuleList | list[nn.Module],
    train_loader: DataLoader,
    *,
    val_loader: DataLoader | None = None,
    alpha: float = 0.95,
    epochs: int = 10,
    optimizer_factory: OptimizerFactory = _DEFAULT_OPTIMIZER_FACTORY,
    scheduler_factory: SchedulerFactory | None = None,
    device: torch.device | str | None = None,
    on_epoch: EpochHook | None = None,
) -> nn.ModuleList | list[nn.Module]:
    """Train a credal relative likelihood ensemble based on :cite:`lohrCredalPrediction2025`.

    Member 0 is the maximum-likelihood reference; each remaining member trains with cross-entropy only until its
    relative likelihood ``exp(ll - max_ll)`` reaches its target, with targets uniform over ``[alpha, 1)``.

    Args:
        predictor: The ``credal_relative_likelihood`` ensemble; members are trained in place.
        train_loader: Loader yielding ``(inputs, targets)`` batches.
        val_loader: Optional validation loader; adds the member's cross-entropy ``"val_loss"`` to the metrics.
        alpha: Lowest relative-likelihood target, in (0, 1]. Default is 0.95; the probly_benchmark CIFAR-10
            config uses 1.0.
        epochs: Maximum number of epochs per member. Default is 10.
        optimizer_factory: Optimizer factory applied per member. Default follows the probly_benchmark CIFAR-10
            recipe: SGD with learning rate 0.1, momentum 0.9, weight decay 5e-4. Weak optimizers leave members
            stuck in the saturated class-biased initialization, see ``tobias_value``.
        scheduler_factory: Scheduler factory applied per member, stepped once per epoch.
        device: If given, move each member and every batch to this device.
        on_epoch: Per-epoch hook receiving the member metrics, plus ``"threshold"`` and ``"relative_likelihood"``
            for members past the reference; returning True stops that member early.

    Returns:
        The trained predictor.

    Raises:
        ValueError: If alpha is outside (0, 1].
    """
    members = list(predictor)
    thresholds = relative_likelihood_thresholds(alpha, len(members))

    reference = members[0]
    train_model(
        reference,
        train_loader,
        F.cross_entropy,
        val_loader=val_loader,
        epochs=epochs,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        device=device,
        on_epoch=on_epoch,
        extra_metrics={"member": 0.0},
    )
    max_ll = _mean_log_likelihood(reference, train_loader, device)

    for i, (member, threshold) in enumerate(zip(members[1:], thresholds, strict=True), start=1):

        def rl_hook(metrics: dict[str, float], member: nn.Module = member, threshold: float = threshold) -> bool:
            relative_likelihood = math.exp(_mean_log_likelihood(member, train_loader, device) - max_ll)
            user_stop = on_epoch({**metrics, "relative_likelihood": relative_likelihood}) if on_epoch else None
            return relative_likelihood >= threshold or bool(user_stop)

        train_model(
            member,
            train_loader,
            F.cross_entropy,
            val_loader=val_loader,
            epochs=epochs,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            device=device,
            on_epoch=rl_hook,
            extra_metrics={"member": float(i), "threshold": threshold},
        )
    return predictor


__all__ = ["train_credal_relative_likelihood_torch"]
