"""Torch training function for the Bayesian method."""

from __future__ import annotations

from collections.abc import Sized
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence
from probly.train.torch import train_model

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.utils.data import DataLoader

    from probly.method.bayesian._common import BayesianPredictor
    from probly.train.torch import EpochHook, OptimizerFactory, SchedulerFactory

# The probly_benchmark CIFAR-10 recipe for Bayesian models: no weight decay, the KL term regularizes.
_DEFAULT_OPTIMIZER_FACTORY = partial(torch.optim.SGD, lr=0.1, momentum=0.9)


def train_bayesian[**In, Out](
    predictor: BayesianPredictor[In, Out],
    train_loader: DataLoader,
    *,
    val_loader: DataLoader | None = None,
    kl_scale: float = 1.0,
    dataset_size: int | None = None,
    epochs: int = 10,
    optimizer_factory: OptimizerFactory = _DEFAULT_OPTIMIZER_FACTORY,
    scheduler_factory: SchedulerFactory | None = None,
    device: torch.device | str | None = None,
    on_epoch: EpochHook | None = None,
) -> BayesianPredictor[In, Out]:
    """Train a Bayesian predictor on the ELBO of :cite:`blundellWeightUncertainty2015`.

    Every forward pass samples weights from the variational posterior, so minimizing cross-entropy plus
    ``kl_scale / len(dataset)`` times the summed layer KL divergence is the standard ELBO.

    Args:
        predictor: The ``bayesian`` predictor, trained in place.
        train_loader: Loader yielding ``(inputs, targets)`` batches.
        val_loader: Optional validation loader; adds ``"val_loss"`` (same ELBO objective, a single
            Monte Carlo weight sample per epoch).
        kl_scale: Scale of the KL term; 1 is the exact ELBO, smaller values temper the posterior. Default is 1.0.
        dataset_size: Training-set size ``N`` in the KL penalty ``kl_scale / N``. Inferred from
            ``len(train_loader.dataset)`` when None; pass it explicitly for length-less datasets.
        epochs: Maximum number of epochs. Default is 10.
        optimizer_factory: Optimizer factory. Default follows the probly_benchmark CIFAR-10 recipe for Bayesian
            models: SGD with learning rate 0.1, momentum 0.9, and no weight decay (the KL term already
            regularizes toward the prior).
        scheduler_factory: Scheduler factory, stepped once per epoch.
        device: If given, move the predictor and every batch to this device.
        on_epoch: Per-epoch hook receiving ``{"epoch": ..., "running_loss": ...}``; returning True stops early.

    Returns:
        The trained predictor.

    Raises:
        TypeError: If dataset_size is None and the loader's dataset has no length.
        ValueError: If dataset_size is given but not positive.
    """
    if dataset_size is None:
        dataset = train_loader.dataset
        if not isinstance(dataset, Sized):
            msg = "train_bayesian cannot infer the dataset size; pass dataset_size for length-less datasets."
            raise TypeError(msg)
        dataset_size = len(dataset)
    elif dataset_size < 1:
        msg = f"dataset_size must be >= 1, got {dataset_size}."
        raise ValueError(msg)
    elbo = ELBOLoss(kl_penalty=kl_scale / dataset_size)
    module = cast("nn.Module", predictor)

    def loss_fn(output: Tensor, targets: Tensor) -> Tensor:
        return elbo(output, targets, collect_kl_divergence(module))

    train_model(
        module,
        train_loader,
        loss_fn,
        val_loader=val_loader,
        epochs=epochs,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        device=device,
        on_epoch=on_epoch,
    )
    return predictor


__all__ = ["train_bayesian"]
