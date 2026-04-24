"""Shared efficient credal prediction implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.method import predictor_transformation
from probly.predictor import LogitClassifier, RepresentationPredictor, predict, predict_raw
from probly.representation.credal_set import ProbabilityIntervalsCredalSet, create_probability_intervals_from_bounds

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.array_like import ArrayLike


@runtime_checkable
class EfficientCredalPredictor[**In, Out: ProbabilityIntervalsCredalSet](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the efficient credal prediction method."""

    predictor: Predictor
    lower: ArrayLike[float]
    upper: ArrayLike[float]

    @property
    def lower_bounds(self) -> ArrayLike[float]:
        return self.lower

    @property
    def upper_bounds(self) -> ArrayLike[float]:
        return self.upper


@flexdispatch
def efficient_credal_prediction_generator[**In, Out: ProbabilityIntervalsCredalSet](
    base: Predictor,
    num_classes: int,
) -> EfficientCredalPredictor[In, Out]:
    """Generate an efficient credal predictor from a base model."""
    msg = f"No efficient credal prediction generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@EfficientCredalPredictor.register_factory
def efficient_credal_prediction[**In, Out: ProbabilityIntervalsCredalSet](
    base: Predictor,
    num_classes: int,
) -> EfficientCredalPredictor[In, Out]:
    """Create an efficient credal predictor from a base predictor based on :cite:`hofmanefficient`.

    Args:
        base: Predictor, The base model to be used for the efficient credal predictor.
        num_classes: int, The number of classes to predict.

    Returns:
        Predictor, The efficient credal predictor.
    """
    return efficient_credal_prediction_generator(base, num_classes)


@predict.register(EfficientCredalPredictor)
def _[**In](
    predictor: EfficientCredalPredictor[In, ProbabilityIntervalsCredalSet], *args: In.args, **kwargs: In.kwargs
) -> ProbabilityIntervalsCredalSet:
    """Predict with a efficient credal predictor."""
    return create_probability_intervals_from_bounds(
        predict_raw(predictor, *args, **kwargs), predictor.lower_bounds, predictor.upper_bounds
    )
