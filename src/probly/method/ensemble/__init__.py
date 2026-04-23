"""Ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import (
    FLAX_LIST,
    FLAX_MODULE,
    RIVER_ARF,
    RIVER_ARF_REGRESSOR,
    SKLEARN_MODULE,
    TORCH_MODULE,
    TORCH_MODULE_LIST,
)
from probly.predictor import predict_raw

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

EnsembleCategoricalDistributionPredictor.register(RIVER_ARF)
EnsemblePredictor.register(RIVER_ARF_REGRESSOR)


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


## River
@predict_raw.delayed_register(RIVER_ARF)
def _(_: type) -> None:
    from . import river as river  # noqa: PLC0415


@predict_raw.delayed_register(RIVER_ARF_REGRESSOR)
def _(_: type) -> None:
    from . import river as river  # noqa: PLC0415


__all__ = [
    "EnsembleCategoricalDistributionPredictor",
    "EnsembleDirichletDistributionPredictor",
    "EnsemblePredictor",
    "ensemble",
    "ensemble_generator",
    "register_ensemble_members",
]
