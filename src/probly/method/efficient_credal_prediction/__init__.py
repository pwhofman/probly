"""Efficient credal prediction method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import EfficientCredalPredictor, efficient_credal_prediction, efficient_credal_prediction_generator


## Torch
@efficient_credal_prediction_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "EfficientCredalPredictor",
    "efficient_credal_prediction",
]
