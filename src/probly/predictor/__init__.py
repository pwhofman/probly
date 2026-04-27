"""Module for predictors."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, SKLEARN_MODULE, TORCH_MODULE

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
    predict_single,
    predictor_registry,
    to_single_prediction,
)

# Aliases for common predictor types
ProbabilisticClassifier = CategoricalDistributionPredictor
LogitClassifier = LogitDistributionPredictor
EvidentialPredictor = DirichletDistributionPredictor

# Register common module types as predictors
Predictor.register(
    (
        TORCH_MODULE,
        FLAX_MODULE,
    )
)


@predict_raw.delayed_register(SKLEARN_MODULE)
def _(_: type[object]) -> None:
    from . import sklearn as sklearn  # noqa: PLC0415


__all__ = [
    "CategoricalDistributionPredictor",
    "CredalPredictor",
    "DirichletDistributionPredictor",
    "DistributionPredictor",
    "EvidentialPredictor",
    "IterablePredictor",
    "LogitClassifier",
    "LogitDistributionPredictor",
    "Predictor",
    "PredictorName",
    "ProbabilisticClassifier",
    "RandomPredictor",
    "RepresentationPredictor",
    "predict",
    "predict_raw",
    "predict_single",
    "predictor_registry",
    "to_single_prediction",
]
