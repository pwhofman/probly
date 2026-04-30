"""Shared credal-bounds transformation implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.predictor import LogitClassifier
from probly.representation.distribution import CategoricalDistribution
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.array_like import ArrayLike


@runtime_checkable
class CredalBoundsPredictor[**In, Out: CategoricalDistribution](LogitClassifier[In, Out], Protocol):
    """Logit classifier wrapped with calibrated lower and upper bounds.

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
def credal_bounds_generator[**In, Out: CategoricalDistribution](
    base: Predictor,
) -> CredalBoundsPredictor[In, Out]:
    """Generate a credal-bounds predictor from a base model."""
    msg = f"No credal-bounds generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@CredalBoundsPredictor.register_factory
def credal_bounds[**In, Out: CategoricalDistribution](
    base: Predictor,
) -> CredalBoundsPredictor[In, Out]:
    """Create a predictor that stores calibrated lower and upper logit bounds.

    Args:
        base: The base ``LogitClassifier`` to wrap.

    Returns:
        The credal-bounds predictor; ``predict`` returns the base's
        :class:`CategoricalDistribution`. Use ``representer(...)`` to get the
        credal-set view that combines the distribution with the calibrated
        bounds.
    """
    return credal_bounds_generator(base)
