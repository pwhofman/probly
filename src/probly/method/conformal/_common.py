
from typing import Protocol, runtime_checkable

from probly.calibrator._common import ConformalCalibrator
from probly.conformal.scores._common import ClassificationNonConformityScore, QuantileNonConformityScore, RegressionNonConformityScore
from probly.predictor._common import Predictor


@runtime_checkable
class ConformalClassificationCalibrator[**In, Out](ConformalCalibrator[In, Out], Protocol):
    """A conformal calibrator for classification predictors."""

    conformal_quantile: float
    non_conformity_score: ClassificationNonConformityScore

@runtime_checkable
class ConformalQuantileRegressionCalibrator[**In, Out](ConformalCalibrator[In, Out], Protocol):
    """A conformal calibrator for quantile regression predictors."""

    conformal_quantile: float
    non_conformity_score: QuantileNonConformityScore

@runtime_checkable
class ConformalRegressionCalibrator[**In, Out](ConformalCalibrator[In, Out], Protocol):
    """A conformal calibrator for regression predictors."""

    conformal_quantile: float
    non_conformity_score: RegressionNonConformityScore

def conformal_generator[**In, Out](model: Predictor[In, Out]) -> ConformalCalibrator[In, Out]:
    """Generate a conformal predictor from a base model."""
    object.__setattr__(model, "conformal_quantile", None)
    object.__setattr__(model, "non_conformity_score", None)
    return model


@ConformalClassificationCalibrator.register_factory
def conformalize_classifier[**In, Out](model: Predictor[In, Out]) -> ConformalClassificationCalibrator[In, Out]:
    """Conformalise a classification predictor.

    This factory function creates a conformal predictor from a base classification model.

    Args:
        model: A base classification predictor to be conformalized.

    Returns:
        A conformal classification calibrator that can be calibrated using a calibration dataset.

    """
    return conformal_generator(model)



@ConformalQuantileRegressionCalibrator.register_factory
def conformalize_quantile_regressor[**In, Out](
    model: Predictor[In, Out],
) -> ConformalQuantileRegressionCalibrator[In, Out]:
    """Conformalise a quantile regression predictor.

    Args:
        model: A base quantile regression predictor to be conformalized.

    Returns:
        A conformal quantile regression calibrator that can be calibrated using a calibration dataset.

    """
    return conformal_generator(model)


@ConformalRegressionCalibrator.register_factory
def conformalize_regressor[**In, Out](model: Predictor[In, Out]) -> ConformalRegressionCalibrator[In, Out]:
    """Conformalise a regression predictor.

    Args:
        model: A base regression predictor to be conformalized.

    Returns:
        A conformal regression calibrator that can be calibrated using a calibration dataset.

    """
    return conformal_generator(model)
