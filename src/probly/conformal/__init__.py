"""This module contains the implementation of conformal methods in probly."""

from probly.calibrator._common import ConformalCalibrator

from ..method.conformal import conformalize_classifier, conformalize_quantile_regressor, conformalize_regressor
from ..utils.quantile import calculate_quantile, calculate_weighted_quantile
from .scores import AbsoluteErrorScore, APSScore, CQRrScore, CQRScore, LACScore, RAPSScore, SAPSScore, UACQRScore
from .utils import is_conformal_calibrated

__all__ = [
    "APSScore",
    "AbsoluteErrorScore",
    "CQRScore",
    "CQRrScore",
    "ConformalCalibrator",
    "LACScore",
    "RAPSScore",
    "SAPSScore",
    "UACQRScore",
    "calculate_quantile",
    "calculate_weighted_quantile",
    "conformalize_classifier",
    "conformalize_quantile_regressor",
    "conformalize_regressor",
    "is_conformal_calibrated",
]
