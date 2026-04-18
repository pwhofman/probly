"""Conformalized prediction with credal sets as output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from probly.method.conformal_credal_set_prediction.calibrate import calibrate_raw_conformal
from probly.method.conformal_credal_set_prediction.quantile import calculate_quantile
from probly.predictor import CredalPredictor, Predictor, predict
from probly.representation.credal_set import CategoricalCredalSet
from probly.representation.credal_set.array import ArrayDistanceBasedCredalSet
from probly.representation.distribution import ArrayCategoricalDistribution

if TYPE_CHECKING:
    from probly.method.conformal_credal_set_prediction.scores import NonConformityFunction


@runtime_checkable
class ConformalCredalSetPredictor[**In, Out: CategoricalCredalSet](CredalPredictor[In, Out], Protocol):
    """Protocol for predictors that output conformalized credal sets."""

    quantile: float | None
    non_conformity_score: NonConformityFunction | None


class ConformalCredalSet[**In](ConformalCredalSetPredictor[In, CategoricalCredalSet]):
    """Concrete implementation of a ConformalCredalSetPredictor."""

    def __init__(self, model: Predictor[In, Any], k_classes: int) -> None:
        """Initialize the predictor.

        Args:
            model: The base model to wrap.
            k_classes: Number of classes.
        """
        self.model = model
        self.k = k_classes
        self.quantile: float | None = None
        self.non_conformity_score: NonConformityFunction | None = None

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> ArrayDistanceBasedCredalSet:
        """Predict the credal set for the given inputs.

        Returns:
            The predicted credal set.

        Raises:
            RuntimeError: If the predictor is not calibrated.
        """
        if self.quantile is None:
            msg = "Predictor must be calibrated before calling predict()."
            raise RuntimeError(msg)

        preds = predict(self.model, *args, **kwargs)
        probs = np.asarray(preds)

        return ArrayDistanceBasedCredalSet(
            nominal=ArrayCategoricalDistribution(probs),
            radius=np.full(len(probs), self.quantile),
        )

    @property
    def is_calibrated(self) -> bool:
        """Check if the predictor is calibrated."""
        return self.quantile is not None


def conformal_credal_set_generator[**In, Out: CategoricalCredalSet](
    model: Predictor[In, Any], k_classes: int
) -> ConformalCredalSetPredictor[In, Out]:
    """Generate a conformal credal set predictor form a base model.

    Args:
        model: The base model.
        k_classes: Number of classes.

    Returns:
        The generated predictor.
    """
    return ConformalCredalSet(model, k_classes=k_classes)  # type: ignore[return-value]


@ConformalCredalSetPredictor.register_factory
def conformal_credal_set_prediction[**In, Out: CategoricalCredalSet](
    model: Predictor[In, Any], k_classes: int
) -> ConformalCredalSetPredictor[In, Out]:
    """Create a conformalized credal set predictor.

    Args:
        model: The base model.
        k_classes: Number of classes.

    Returns:
        The conformalized predictor.
    """
    return conformal_credal_set_generator(model, k_classes=k_classes)


@calibrate_raw_conformal.register(ConformalCredalSet)
def conformal_credal_set_calibration[In, Out](
    predictor: ConformalCredalSet[In],
    non_conformity_score: NonConformityFunction,
    x_calib: In,
    y_calib: Out,
    alpha: float,
) -> ConformalCredalSet[In]:
    """Calibrate a conformal credal set predictor."""
    prediction = predict(predictor.model, x_calib)
    probs = np.asarray(prediction)

    y_calib_np = np.asarray(y_calib)
    if y_calib_np.ndim == 1:
        n_samples = len(y_calib_np)
        y_one_hot = np.zeros((n_samples, predictor.k))
        y_one_hot[np.arange(n_samples), y_calib_np.astype(int)] = 1.0
        y_calib_np = y_one_hot

    scores = non_conformity_score(probs, y_calib_np)
    quantile = calculate_quantile(scores, alpha)
    predictor.quantile = quantile
    predictor.non_conformity_score = non_conformity_score
    return predictor
