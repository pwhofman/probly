"""Module for predictors."""

from __future__ import annotations

from ._common import CredalPredictor, DistributionPredictor, IterablePredictor, Predictor, RandomPredictor, predict

__all__ = [
    "CredalPredictor",
    "DistributionPredictor",
    "IterablePredictor",
    "Predictor",
    "RandomPredictor",
    "predict",
]
