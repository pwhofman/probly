"""Conformalized prediction with credal sets as output."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, runtime_checkable

from flextype import flexdispatch
from probly.conformal_scores.dirichlet_relative_likelihood._common import dirichlet_rl_score
from probly.conformal_scores.total_variation._common import tv_score
from probly.method.method import predictor_transformation
from probly.predictor import CredalPredictor, DirichletDistributionPredictor, Predictor, predict
from probly.representation.credal_set._common import (
    DirichletLevelSetCredalSet,
    DistanceBasedCredalSet,
    create_dirichlet_level_set_credal_set,
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


@runtime_checkable
class DirichletConformalCredalSetPredictor[**In, Out: DirichletLevelSetCredalSet](CredalPredictor[In, Out], Protocol):
    """Conformal credal set predictor using Dirichlet relative likelihood.

    Produces instance-adaptive credal sets based on the Dirichlet density
    level set of a second-order predictor.
    """

    predictor: Predictor
    conformal_quantile: float | None
    non_conformity_score: NonConformityScore | None

    def calibrate(self, alpha: float, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the predictor on calibration data."""
        ...


@predict.register(DirichletConformalCredalSetPredictor)
def predict_dirichlet_conformal_credal_set[**In, Out: DirichletLevelSetCredalSet](
    predictor: DirichletConformalCredalSetPredictor[In, Out],
    *args: In.args,
    **kwargs: In.kwargs,
) -> DirichletLevelSetCredalSet:
    """Predict a Dirichlet level set conformal credal set."""
    quantile, _ = calibrated_state(predictor)
    dirichlet_pred = predict(cast("Any", predictor).predictor, *args, **kwargs)
    threshold = 1.0 - quantile
    return create_dirichlet_level_set_credal_set(dirichlet_pred.alphas, threshold)


@predictor_transformation(permitted_predictor_types=(DirichletDistributionPredictor,), preserve_predictor_type=False)
@DirichletConformalCredalSetPredictor.register_factory
def conformal_dirichlet_relative_likelihood[**In, Out: DirichletLevelSetCredalSet](
    base: Predictor[In, Out],
) -> DirichletConformalCredalSetPredictor[In, Out]:
    """Create a conformalized credal set predictor using Dirichlet relative likelihood.

    Wraps a Dirichlet predictor (e.g., from :func:`~probly.method.prior_network.prior_network`)
    and produces instance-adaptive credal sets at prediction time.

    Args:
        base: A predictor that outputs Dirichlet distributions.

    Returns:
        A conformalized credal set predictor.
    """
    return conformal_credal_set_generator(base, dirichlet_rl_score)


__all__ = [
    "ConformalCredalSetPredictor",
    "DirichletConformalCredalSetPredictor",
    "TVConformalCredalSetPredictor",
    "_ConformalCredalSetPredictorBase",
    "conformal_credal_set_generator",
    "conformal_dirichlet_relative_likelihood",
    "conformal_total_variation",
]
