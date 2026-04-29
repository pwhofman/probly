"""Shared efficient credal prediction implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.method import predictor_transformation
from probly.predictor import LogitClassifier
from probly.representation.distribution import CategoricalDistribution

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.array_like import ArrayLike


@runtime_checkable
class EfficientCredalPredictor[**In, Out: CategoricalDistribution](LogitClassifier[In, Out], Protocol):
    """Logit classifier wrapped with calibrated lower/upper bounds, based on :cite:`hofmanefficient`.

    Wraps a base :class:`LogitClassifier` together with externally-calibrated
    per-class lower and upper bound offsets. ``predict`` returns the base's
    :class:`CategoricalDistribution`; the credal-set view (combining the
    distribution with the bounds) is available via the registered
    representer.
    """

    predictor: Predictor
    lower: ArrayLike[float]
    upper: ArrayLike[float]


@flexdispatch
def efficient_credal_prediction_generator[**In, Out: CategoricalDistribution](
    base: Predictor,
) -> EfficientCredalPredictor[In, Out]:
    """Generate an efficient credal predictor from a base model."""
    msg = f"No efficient credal prediction generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@EfficientCredalPredictor.register_factory
def efficient_credal_prediction[**In, Out: CategoricalDistribution](
    base: Predictor,
) -> EfficientCredalPredictor[In, Out]:
    """Create an efficient credal predictor from a base predictor based on :cite:`hofmanefficient`.

    Args:
        base: The base ``LogitClassifier`` to wrap.

    Returns:
        The efficient credal predictor; ``predict`` returns the base's
        :class:`CategoricalDistribution`. Use ``representer(...)`` to get the
        credal-set view that combines the distribution with the calibrated
        bounds.
    """
    return efficient_credal_prediction_generator(base)
