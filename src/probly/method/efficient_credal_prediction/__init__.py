"""Efficient credal prediction method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR

from . import numpy as numpy
from ._common import (
    EfficientCredalPredictor,
    compute_efficient_credal_prediction_bounds,
    efficient_credal_prediction,
    efficient_credal_prediction_generator,
)


## Torch
@efficient_credal_prediction_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@compute_efficient_credal_prediction_bounds.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "EfficientCredalPredictor",
    "compute_efficient_credal_prediction_bounds",
    "efficient_credal_prediction",
    "efficient_credal_prediction_generator",
]
