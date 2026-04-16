"""Conformal prediction transformer methods."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.calibrator._common import ConformalCalibrator, calibrate_raw_conformal
from probly.predictor import ProbabilisticClassifier
from probly.predictor._common import (
    DistributionPredictor,
    Predictor,
    predict_raw,
)
from probly.representation.distribution._common import CategoricalDistribution
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.sample import create_sample
from probly.representation.sample.array import ArraySample
from probly.utils.quantile._common import calculate_quantile

if TYPE_CHECKING:
    from probly.conformal_scores._common import (
        ClassificationNonConformityScore,
        NonConformityScore,
        QuantileNonConformityScore,
        RegressionNonConformityScore,
    )
    from probly.representation.sample._common import Sample


@runtime_checkable  # ty: ignore[conflicting-metaclass]
class ConformalClassificationCalibrator[**In, Out: CategoricalDistribution](
    ConformalCalibrator[In, Out],
    ProbabilisticClassifier[In, Out],
    Protocol,
):
    """A conformal calibrator for classification predictors."""

    _running_instancehook: ClassVar[ContextVar[object]] = ContextVar(
        "ConformalClassificationCalibrator._running_instancehook", default=NotImplementedError
    )

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        if cls._running_instancehook.get() is instance:
            return NotImplemented
        try:
            tok = cls._running_instancehook.set(instance)
            # Honor explicit routing from conformalize_quantile_regressor/conformalize_regressor.
            if isinstance(instance, (ConformalQuantileRegressionCalibrator, ConformalRegressionCalibrator)):
                return False
            if isinstance(instance, ConformalCalibrator) and isinstance(instance, ProbabilisticClassifier):
                return True
        finally:
            cls._running_instancehook.reset(tok)
        return NotImplemented

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


def conformal_generator[**In, Out](model: Predictor[In, Out]) -> Predictor[In, Out]:
    """Generate a conformal predictor from a base model."""
    object.__setattr__(model, "conformal_quantile", None)
    object.__setattr__(model, "non_conformity_score", None)
    return model


@ConformalClassificationCalibrator.register_factory
def conformalize_classifier[**In, Out: CategoricalDistribution](
    model: Predictor[In, Out],
) -> ConformalClassificationCalibrator[In, Out]:
    """Conformalise a classification predictor.

    This factory function creates a conformal predictor from a base classification model.

    Args:
        model: A base classification predictor to be conformalized.

    Returns:
        A conformal classification calibrator that can be calibrated using a calibration dataset.

    """
    return conformal_generator(model)  # ty: ignore[invalid-return-type]


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
    return conformal_generator(model)  # ty: ignore[invalid-return-type]


@ConformalRegressionCalibrator.register_factory
def conformalize_regressor[**In, Out](model: Predictor[In, Out]) -> ConformalRegressionCalibrator[In, Out]:
    """Conformalise a regression predictor.

    Args:
        model: A base regression predictor to be conformalized.

    Returns:
        A conformal regression calibrator that can be calibrated using a calibration dataset.

    """
    return conformal_generator(model)  # ty: ignore[invalid-return-type]


@lazydispatch
def ensure_distribution_2d(prediction: Sample[Any]) -> DistributionPredictor:
    """Ensure that the prediction is a distribution predictor.

    Aggregate over the sample dimension if the prediction has more than 2 dimensions.
    """
    msg = f"Cannot ensure distribution for prediction of type {type(prediction)}."
    raise NotImplementedError(msg)


@ensure_distribution_2d.register(ArraySample)
def _(prediction: ArraySample[Any]) -> DistributionPredictor:
    """Ensure that the prediction is a distribution predictor.

    Aggregate over the sample dimension if the prediction has more than 2 dimensions.
    """
    prediction = prediction.move_sample_axis(0)
    data_a = prediction.array
    if data_a.ndim > 3:
        msg = "Predictions with more than 3 dimensions are not supported for conformal classification."
        raise ValueError(msg)
    if isinstance(data_a, ArrayCategoricalDistribution):
        return data_a
    if data_a.ndim < 2:
        msg = "The predictor must return a distribution to be conformalized."
        raise ValueError(msg)
    if not np.allclose(data_a.sum(axis=-1), np.ones_like(data_a[..., 0])):
        data_a = np.exp(data_a - np.max(data_a, axis=-1, keepdims=True))
        data_a = data_a / data_a.sum(axis=-1, keepdims=True)
    if data_a.ndim == 3:
        data_a = data_a.mean(axis=0)
    return ArrayCategoricalDistribution(unnormalized_probabilities=data_a)


@calibrate_raw_conformal.register(ConformalClassificationCalibrator)
def conformal_classification_calibration[In, Out](
    predictor: ConformalClassificationCalibrator,
    non_conformity_score: ClassificationNonConformityScore,
    x_calib: In,
    y_calib: Out,
    alpha: float,
) -> ConformalClassificationCalibrator:
    """Calibrate a conformal predictor."""
    prediction = create_sample(predict_raw(predictor, x_calib), sample_axis=0)
    probability = ensure_distribution_2d(prediction)
    scores = non_conformity_score(probability, y_calib)
    quantile = calculate_quantile(scores, alpha)
    predictor.conformal_quantile = quantile
    predictor.non_conformity_score = non_conformity_score
    return predictor


@calibrate_raw_conformal.register(ConformalRegressionCalibrator)
@calibrate_raw_conformal.register(ConformalQuantileRegressionCalibrator)
def conformal_regression_calibration[In, Out](
    predictor: ConformalRegressionCalibrator | ConformalQuantileRegressionCalibrator,
    non_conformity_score: NonConformityScore,
    x_calib: In,
    y_calib: Out,
    alpha: float,
) -> ConformalRegressionCalibrator | ConformalQuantileRegressionCalibrator:
    """Calibrate a conformal predictor."""
    prediction = create_sample(predict_raw(predictor, x_calib), sample_axis=0)
    scores = non_conformity_score(prediction, y_calib)
    quantile = calculate_quantile(scores, alpha)
    predictor.conformal_quantile = quantile
    predictor.non_conformity_score = cast("Any", non_conformity_score)
    return predictor
