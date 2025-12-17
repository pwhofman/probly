"""Ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, SKLEARN_MODULE, TORCH_MODULE

from . import common

ensemble = common.ensemble
register = common.register


## Torch
@common.ensemble_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@common.ensemble_generator.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


## Sklearn
@common.ensemble_generator.delayed_register(SKLEARN_MODULE)
def _(_: type) -> None:
    from . import sklearn_models as sklearn_models  # noqa: PLC0415
