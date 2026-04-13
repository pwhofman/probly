"""Shared dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from lazy_dispatch import lazydispatch
from probly.calibrator._common import ConformalCalibrator, calibrate_raw
from probly.conformal.quantile._common import calculate_quantile
from probly.predictor._common import predict_raw

if TYPE_CHECKING:
    from probly.conformal.scores._common import RegressionNonConformityScore
    from probly.predictor import Predictor


class ConformalRegressionCalibrator[**In, Out](ConformalCalibrator[In, Out], Protocol):
    """A conformal calibrator for regression predictors."""

    conformal_quantile: float
    non_conformity_score: RegressionNonConformityScore


@lazydispatch
def conformal_generator[**In, Out](model: Predictor[In, Out]) -> ConformalRegressionCalibrator[In, Out]:
    """Generate a conformal predictor from a base model."""
    msg = f"No conformal generator is registered for type {type(model)}"
    raise NotImplementedError(msg)


@ConformalRegressionCalibrator.register_factory
def conformalize_regressor[**In, Out](model: Predictor[In, Out]) -> ConformalRegressionCalibrator[In, Out]:
    """Conformalise a predictor."""
    return conformal_generator(model)


@calibrate_raw.register(ConformalRegressionCalibrator)
def conformal_reg_calibration[In, Out](
    predictor: ConformalRegressionCalibrator,
    x_calib: In,
    y_calib: Out,
    non_conformity_score: RegressionNonConformityScore,
    alpha: float,
) -> ConformalRegressionCalibrator:
    """Calibrate a conformal predictor."""
    prediction = predict_raw(predictor, x_calib)
    scores = non_conformity_score(prediction, y_calib)
    quantile = calculate_quantile(scores, alpha)
    predictor.conformal_quantile = quantile
    predictor.non_conformity_score = non_conformity_score
    return predictor
