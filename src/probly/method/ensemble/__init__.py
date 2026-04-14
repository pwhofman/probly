"""Ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_LIST, FLAX_MODULE, SKLEARN_MODULE, TORCH_MODULE, TORCH_MODULE_LIST

from ._common import (
    EnsembleCategoricalDistributionPredictor,
    EnsembleDirichletDistributionPredictor,
    EnsemblePredictor,
    ensemble,
    ensemble_generator,
    register_ensemble_members,
)

EnsemblePredictor.register(
    (
        TORCH_MODULE_LIST,
        FLAX_LIST,
    )
)


## Torch
@ensemble_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@ensemble_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@ensemble_generator.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


## Sklearn
@ensemble_generator.delayed_register(SKLEARN_MODULE)
def _(_: type) -> None:
    from . import sklearn as sklearn  # noqa: PLC0415


__all__ = [
    "EnsembleCategoricalDistributionPredictor",
    "EnsembleDirichletDistributionPredictor",
    "EnsemblePredictor",
    "ensemble",
    "ensemble_generator",
    "register_ensemble_members",
]
