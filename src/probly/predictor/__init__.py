"""Module for predictors."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import (
    CategoricalDistributionPredictor,
    CredalPredictor,
    DistributionPredictor,
    IterablePredictor,
    LogitDistributionPredictor,
    Predictor,
    PredictorName,
    RandomPredictor,
    predict,
    predict_raw,
    predictor_registry,
)

# Aliases for common predictor types
ProbabilisticClassifier = CategoricalDistributionPredictor
LogitClassifier = LogitDistributionPredictor

# Register common module types as predictors
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
    "LogitClassifier",
    "Predictor",
    "PredictorName",
    "ProbabilisticClassifier",
    "RandomPredictor",
    "predict",
    "predict_raw",
    "predictor_registry",
]
