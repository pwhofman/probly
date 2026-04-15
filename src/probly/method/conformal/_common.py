"""Conformal prediction transformer methods."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.calibrator._common import ConformalCalibrator, calibrate_raw, calibrate_raw_conformal
from probly.conformal_scores._common import NonConformityScore
from probly.method.method import predictor_transformation
from probly.predictor import ProbabilisticClassifier
from probly.predictor._common import Predictor, predict, predict_raw
from probly.representation.sample import create_sample
from probly.utils.quantile._common import calculate_quantile

if TYPE_CHECKING:
    from probly.conformal_scores._common import (
        ClassificationNonConformityScore,
        QuantileNonConformityScore,
        RegressionNonConformityScore,
    )


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


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier,),
    preserve_predictor_type=True,
)
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


@calibrate_raw_conformal.register(ConformalCalibrator)
@predictor_transformation(
    permitted_predictor_types=(
        ConformalClassificationCalibrator,
        ConformalQuantileRegressionCalibrator,
        ConformalRegressionCalibrator,
    ),
    preserve_predictor_type=True,
)
def conformal_classification_calibration[In, Out](
    predictor: ConformalCalibrator,
    non_conformity_score: NonConformityScore,
    x_calib: In,
    y_calib: Out,
    alpha: float,
) -> ConformalClassificationCalibrator:
    """Calibrate a conformal predictor."""
    prediction = create_sample(predict(predictor, x_calib), sample_axis=0)
    print(prediction)
    scores = non_conformity_score(prediction, y_calib)
    quantile = calculate_quantile(scores, alpha)
    # "Delete" the existing types, to make the representer not complain about ambiguous dispatch.
    # This is a bit hacky, but needed as we don't have super() support for the dispatching system
    copied_predictor = copy.deepcopy(predictor)
    copied_predictor.conformal_quantile = quantile
    copied_predictor.non_conformity_score = non_conformity_score
    return copied_predictor
