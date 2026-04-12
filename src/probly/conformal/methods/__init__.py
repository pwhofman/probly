"""Conformal transformations for regression and classification."""

from __future__ import annotations

from .classification import ConformalClassificationCalibrator, conformalize_classifier
from .quantile_regression import ConformalQuantileRegressionCalibrator, conformalize_quantile_regressor
from .regression import ConformalRegressionCalibrator, conformalize_regressor

__all__ = [
    "ConformalClassificationCalibrator",
    "ConformalQuantileRegressionCalibrator",
    "ConformalRegressionCalibrator",
    "conformalize_classifier",
    "conformalize_quantile_regressor",
    "conformalize_regressor",
]
