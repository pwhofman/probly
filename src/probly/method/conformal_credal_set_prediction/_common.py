"""Conformalized prediction with credal sets as output."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, runtime_checkable

from flextype import flexdispatch
from probly.conformal_scores.total_variation._common import tv_score
from probly.method.method import predictor_transformation
from probly.predictor import CredalPredictor, Predictor, predict
from probly.representation.credal_set._common import (
    DistanceBasedCredalSet,
    create_distance_based_credal_set_from_center_and_radius,
)
from probly.utils.quantile._common import calculate_quantile

if TYPE_CHECKING:
    from probly.conformal_scores._common import NonConformityScore


@runtime_checkable
class ConformalCredalSetPredictor[**In, Out: DistanceBasedCredalSet](CredalPredictor[In, Out], Protocol):
    """Protocol for predictors that output conformalized credal sets."""

    predictor: Predictor
    conformal_quantile: float | None
    non_conformity_score: NonConformityScore | None

    def calibrate(self, alpha: float, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the predictor on calibration data."""
        ...


@runtime_checkable
class TVConformalCredalSetPredictor[**In, Out: DistanceBasedCredalSet](ConformalCredalSetPredictor[In, Out], Protocol):
    """Conformal Credal Set predictor for TV."""


class _ConformalCredalSetPredictorBase[**In, Out](ABC):
    """Concrete implementation of a ConformalCredalSetPredictor."""

    predictor: Predictor[In, Out]
    conformal_quantile: float | None
    non_conformity_score: NonConformityScore[Out, Out] | None

    def __init__(self, predictor: Predictor[In, Out], non_conformity_score: NonConformityScore[Out, Out]) -> None:
        super().__init__()
        self.predictor = predictor
        self.conformal_quantile = None
        self.non_conformity_score = non_conformity_score

    def _require_score(self) -> NonConformityScore[Out, Out]:
        if self.non_conformity_score is None:
            msg = "No non_conformity_score configured for this conformal predictor."
            raise ValueError(msg)
        return self.non_conformity_score

    def _require_calibrated(self) -> tuple[float, NonConformityScore[Out, Out]]:
        quantile = self.conformal_quantile
        if quantile is None:
            msg = "Conformal predictor is not calibrated. Please call calibrate() before prediction."
            raise ValueError(msg)
        score = self._require_score()
        return quantile, score

    def calibrate(self, alpha: float, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the predictor using calibration data."""
        score = self._require_score()
        prediction = predict(self.predictor, *calib_args, **calib_kwargs)
        scores = score(prediction, y_calib)
        self.conformal_quantile = calculate_quantile(scores, alpha)
        return self


@flexdispatch
def calibrated_state(_: object) -> tuple[float, NonConformityScore[Any, Any]]:
    msg = "Predictor is not a conformal predictor or is not calibrated."
    raise ValueError(msg)


@calibrated_state.register(_ConformalCredalSetPredictorBase)
def _[**In, Out](
    predictor: _ConformalCredalSetPredictorBase[In, Out],
) -> tuple[float, NonConformityScore[Out, Out]]:
    return predictor._require_calibrated()  # noqa: SLF001


@predict.register(TVConformalCredalSetPredictor)
def predict_total_variation_conformal_credal_set[**In, Out: DistanceBasedCredalSet](
    predictor: TVConformalCredalSetPredictor[In, Out],
    *args: In.args,
    **kwargs: In.kwargs,
) -> DistanceBasedCredalSet:
    """Predict a total variation conformal credal set."""
    quantile, _ = calibrated_state(predictor)
    prediction = predict(cast("Any", predictor).predictor, *args, **kwargs)
    return create_distance_based_credal_set_from_center_and_radius(prediction, quantile)


@flexdispatch
def conformal_credal_set_generator[**In, T, Out](
    base: Predictor[In, Out],
    non_conformity_score: NonConformityScore[Out, T],
) -> ConformalCredalSetPredictor[In, DistanceBasedCredalSet]:
    """Generate a backend-specific conformal set predictor wrapper."""
    msg = f"No conformal generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@TVConformalCredalSetPredictor.register_factory
def conformal_total_variation[**In, Out: DistanceBasedCredalSet](
    base: Predictor[In, Out],
) -> TVConformalCredalSetPredictor[In, Out]:
    return conformal_credal_set_generator(base, tv_score)


__all__ = [
    "ConformalCredalSetPredictor",
    "TVConformalCredalSetPredictor",
    "_ConformalCredalSetPredictorBase",
    "conformal_credal_set_generator",
    "conformal_total_variation",
]
