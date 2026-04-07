"""Module for predictors."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import (
    CategoricalDistributionPredictor,
    CredalPredictor,
    DirichletDistributionPredictor,
    DistributionPredictor,
    IterablePredictor,
    LogitDistributionPredictor,
    Predictor,
    PredictorName,
    RandomPredictor,
    RepresentationPredictor,
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
    "DirichletDistributionPredictor",
    "DistributionPredictor",
    "IterablePredictor",
    "LogitClassifier",
    "Predictor",
    "PredictorName",
    "ProbabilisticClassifier",
    "RandomPredictor",
    "RepresentationPredictor",
    "predict",
    "predict_raw",
    "predictor_registry",
]
