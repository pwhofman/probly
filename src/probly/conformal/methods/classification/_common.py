"""Shared dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from lazy_dispatch import lazydispatch
from probly.calibrator._common import calibrate_raw
from probly.conformal._common import ConformalCalibrator
from probly.conformal.quantile._common import calculate_quantile
from probly.predictor._common import (
    ConformalClassificationPredictor,
    ConformalPredictor,
    predict_raw,
)

if TYPE_CHECKING:
    from probly.conformal.scores._common import ClassificationNonConformityScore
    from probly.predictor import Predictor


@runtime_checkable
class ConformalClassificationCalibrator[**In, Out](ConformalCalibrator[In, Out], Protocol):
    """A conformal calibrator for classification predictors."""

    conformal_quantile: float
    non_conformity_score: ClassificationNonConformityScore


@lazydispatch
def conformal_generator[**In, Out](model: Predictor[In, Out]) -> ConformalClassificationCalibrator[In, Out]:
    """Generate a conformal predictor from a base model."""
    msg = f"No conformal generator is registered for type {type(model)}"
    raise NotImplementedError(msg)


@ConformalClassificationCalibrator.register_factory
@ConformalClassificationPredictor.register_factory
def conformalize_classifier[**In, Out](model: Predictor[In, Out]) -> ConformalClassificationCalibrator[In, Out]:
    """Conformalise a classification predictor.

    This factory function creates a conformal predictor from a base classification model.

    Args:
        model: A base classification predictor to be conformalized.

    Returns:
        A conformal classification calibrator that can be calibrated using a calibration dataset.

    """
    return conformal_generator(model)


@calibrate_raw.register(ConformalClassificationCalibrator)
def conformal_class_calibration[In, Out](
    predictor: ConformalClassificationCalibrator,
    x_calib: In,
    y_calib: Out,
    non_conformity_score: ClassificationNonConformityScore,
    alpha: float,
) -> ConformalPredictor:
    """Calibrate a conformal predictor."""
    prediction = predict_raw(predictor, x_calib)
    probabilities = to_probabilities(prediction)
    scores = non_conformity_score(probabilities, y_calib)
    quantile = calculate_quantile(scores, alpha)
    predictor.conformal_quantile = quantile
    predictor.non_conformity_score = non_conformity_score
    return predictor


@lazydispatch
def to_probabilities[T](predictions: T) -> T:
    """Convert raw model outputs to probabilities."""
    msg = f"No probability conversion function registered for this type of model output. {type(predictions)}"
    raise NotImplementedError(msg)


@to_probabilities.register(np.ndarray)
def _(pred: np.ndarray) -> np.ndarray:
    """Obtain probabilities from a PyTorch model."""
    if pred.ndim != 2:
        msg = f"Probability extraction expects a 2D array, got {pred.ndim}D array instead."
        raise ValueError(msg)
    if np.allclose(pred.sum(axis=-1), 1):
        # If the predictions already sum to 1, we assume they are probabilities
        return pred
    probs = np.exp(pred) / np.sum(np.exp(pred), axis=-1, keepdims=True)
    return probs


@predict_raw.register(ConformalClassificationCalibrator)
def _[**In, Out](predictor: ConformalClassificationCalibrator[In, Out], *args: In.args, **kwargs: In.kwargs) -> Out:
    """Obtain class probabilities from a conformal classification predictor."""
    if hasattr(predictor, "predict_proba"):
        return predictor.predict_proba(*args, **kwargs)  # ty:ignore[call-non-callable]
    if callable(predictor):
        pred = predictor(*args, **kwargs)  # ty:ignore[call-top-callable]
        probs = to_probabilities(pred)
        return probs  # type: ignore[return-value]
    msg = f"Predict function not found for predictor of type {type(predictor)}"
    raise NotImplementedError(msg)
