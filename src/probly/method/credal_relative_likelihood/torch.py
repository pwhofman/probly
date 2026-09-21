"""Torch training function for the credal relative likelihood method."""

from __future__ import annotations

from functools import partial
import math
from typing import TYPE_CHECKING, cast

import torch
from torch.nn import functional as F

from probly.method.credal_relative_likelihood._common import relative_likelihood_thresholds
from probly.train.torch import evaluate_model_mean_loss, train_model

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

    from probly.method.credal_relative_likelihood._common import CredalRelativeLikelihoodPredictor
    from probly.train.torch import EpochHook, OptimizerFactory, SchedulerFactory

# The probly_benchmark CIFAR-10 recipe.
_DEFAULT_OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=5e-4)


def train_credal_relative_likelihood[**In, Out](
    predictor: CredalRelativeLikelihoodPredictor[In, Out],
    train_loader: DataLoader,
    *,
    val_loader: DataLoader | None = None,
    alpha: float = 0.95,
    epochs: int = 10,
    optimizer_factory: OptimizerFactory = _DEFAULT_OPTIMIZER_FACTORY,
    scheduler_factory: SchedulerFactory | None = None,
    device: torch.device | str | None = None,
    on_epoch: EpochHook | None = None,
) -> None:
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

    Raises:
        ValueError: If alpha is outside (0, 1].
    """
    members = [cast("nn.Module", member) for member in predictor]
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
    max_ll = -evaluate_model_mean_loss(reference, train_loader, F.cross_entropy, device=device)

    for i, (member, threshold) in enumerate(zip(members[1:], thresholds, strict=True), start=1):

        def rl_hook(metrics: dict[str, float], member: nn.Module = member, threshold: float = threshold) -> bool:
            ll = -evaluate_model_mean_loss(member, train_loader, F.cross_entropy, device=device)
            relative_likelihood = math.exp(ll - max_ll)
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


__all__ = ["train_credal_relative_likelihood"]
