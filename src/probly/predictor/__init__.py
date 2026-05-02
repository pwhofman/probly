"""Module for predictors."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, LAPLACE_BASE, SKLEARN_MODULE, TORCH_MODULE

from ._common import (
    CategoricalDistributionPredictor,
    CredalPredictor,
    DirichletDistributionPredictor,
    DistributionPredictor,
    GaussianDistributionPredictor,
    IterablePredictor,
    LogitDistributionPredictor,
    Predictor,
    PredictorName,
    RandomPredictor,
    RandomRepresentationPredictor,
    RepresentationPredictor,
    predict,
    predict_raw,
    predictor_registry,
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


@predict.delayed_register(LAPLACE_BASE)
def _(_: type[object]) -> None:
    from . import laplace as laplace  # noqa: PLC0415


__all__ = [
    "CategoricalDistributionPredictor",
    "CredalPredictor",
    "DirichletDistributionPredictor",
    "DistributionPredictor",
    "EvidentialPredictor",
    "GaussianDistributionPredictor",
    "IterablePredictor",
    "LogitClassifier",
    "LogitDistributionPredictor",
    "Predictor",
    "PredictorName",
    "ProbabilisticClassifier",
    "RandomPredictor",
    "RandomRepresentationPredictor",
    "RepresentationPredictor",
    "predict",
    "predict_raw",
    "predictor_registry",
]
