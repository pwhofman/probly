"""Class-bias ensemble transformation."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import ClassBiasEnsemblePredictor, class_bias_ensemble, class_bias_ensemble_traverser


## Torch
@class_bias_ensemble_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@class_bias_ensemble_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = [
    "ClassBiasEnsemblePredictor",
    "class_bias_ensemble",
    "class_bias_ensemble_traverser",
]
