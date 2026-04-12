"""Shared dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from probly.calibrator._common import ConformalCalibrator, calibrate_raw
from probly.conformal.quantile._common import calculate_quantile
from probly.method.ensemble._common import EnsemblePredictor
from probly.method.method import predictor_transformation
from probly.predictor._common import ConformalQuantileRegressionPredictor, predict_raw
from probly.representation.sample import create_sample

if TYPE_CHECKING:
    from probly.conformal.scores._common import QuantileNonConformityScore
    from probly.predictor import Predictor


@runtime_checkable
class ConformalQuantileRegressionCalibrator[**In, Out](ConformalCalibrator[In, Out], Protocol):
    """A conformal calibrator for regression predictors."""

    conformal_quantile: float
    non_conformity_score: QuantileNonConformityScore


@lazydispatch
def conformal_generator[**In, Out](model: Predictor[In, Out]) -> ConformalQuantileRegressionCalibrator[In, Out]:
    """Generate a conformal predictor from a base model."""
    msg = f"No conformal generator is registered for type {type(model)}"
    raise NotImplementedError(msg)


# @predictor_transformation(permitted_predictor_types=(EnsemblePredictor,), preserve_predictor_type=True)
@ConformalQuantileRegressionCalibrator.register_factory
@ConformalQuantileRegressionPredictor.register_factory
def conformalize_quantile_regressor[**In, Out](
    model: Predictor[In, Out],
) -> ConformalQuantileRegressionPredictor[In, Out]:
    """Conformalise a predictor."""
    return conformal_generator(model)


@calibrate_raw.register(ConformalQuantileRegressionCalibrator)
def conformal_reg_calibration[In, Out](
    predictor: ConformalQuantileRegressionCalibrator,
    x_calib: In,
    y_calib: Out,
    non_conformity_score: QuantileNonConformityScore,
    alpha: float,
) -> ConformalQuantileRegressionPredictor[In, Out]:
    """Calibrate a conformal predictor."""
    prediction = create_sample(predict_raw(predictor, x_calib), sample_axis=0)
    scores = non_conformity_score(prediction, y_calib)
    quantile = calculate_quantile(scores, alpha)
    predictor.conformal_quantile = quantile
    predictor.non_conformity_score = non_conformity_score
    return predictor
