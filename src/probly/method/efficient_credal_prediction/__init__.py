"""Efficient credal prediction method."""

from __future__ import annotations

from probly.lazy_types import SKLEARN_MODULE, TORCH_MODULE

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


@compute_efficient_credal_prediction_bounds.delayed_register("torch.Tensor")
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## NumPy
@efficient_credal_prediction_generator.delayed_register(SKLEARN_MODULE)
def _(_: type) -> None:
    from . import numpy as numpy  # noqa: PLC0415


@compute_efficient_credal_prediction_bounds.delayed_register("numpy.ndarray")
def _(_: type) -> None:
    from . import numpy as numpy  # noqa: PLC0415


__all__ = [
    "EfficientCredalPredictor",
    "compute_efficient_credal_prediction_bounds",
    "efficient_credal_prediction",
    "efficient_credal_prediction_generator",
]
