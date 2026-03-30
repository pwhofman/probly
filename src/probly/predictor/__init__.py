"""Module for predictors."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import CredalPredictor, DistributionPredictor, IterablePredictor, Predictor, RandomPredictor, predict

Predictor.register(
    (
        TORCH_MODULE,
        FLAX_MODULE,
    )
)

__all__ = [
    "CredalPredictor",
    "DistributionPredictor",
    "IterablePredictor",
    "Predictor",
    "RandomPredictor",
    "predict",
]
