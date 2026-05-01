"""Shared efficient credal prediction implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

from flextype import flexdispatch
from probly.predictor import LogitClassifier, predict_raw
from probly.representation.array_like import ArrayLike
from probly.representation.credal_set import (
    ProbabilityIntervalsCredalSet,
    create_probability_intervals_from_lower_upper_array,
)
from probly.representation.distribution import CategoricalDistribution
from probly.representer import Representer, representer
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class EfficientCredalPredictor[**In, Out: CategoricalDistribution](LogitClassifier[In, Out], Protocol):
    """Logit classifier wrapped with calibrated lower and upper logit offsets."""

    predictor: Predictor
    lower: ArrayLike[float]
    upper: ArrayLike[float]


@flexdispatch
def efficient_credal_prediction_generator[**In, Out: CategoricalDistribution](
    base: Predictor,
) -> EfficientCredalPredictor[In, Out]:
    """Generate an efficient credal prediction predictor from a base model."""
    msg = f"No efficient credal prediction generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@EfficientCredalPredictor.register_factory
def efficient_credal_prediction[**In, Out: CategoricalDistribution](
    base: Predictor,
) -> EfficientCredalPredictor[In, Out]:
    """Create a predictor that stores calibrated lower and upper logit offsets.

    Args:
        base: The base logit classifier to wrap.

    Returns:
        The efficient credal prediction predictor. Use ``representer(...)`` to get
        the credal-set view that combines logits with the calibrated offsets.
    """
    return efficient_credal_prediction_generator(base)


@flexdispatch
def compute_efficient_credal_bounds[T: ArrayLike](logits: T, lower: T, upper: T) -> T:
    """Compute packed interval probability bounds via 2K logit perturbations.

    Args:
        logits: Base classifier output of shape ``(B, C)``.
        lower: Per-class signed lower-direction logit offset, shape ``(C,)``.
        upper: Per-class signed upper-direction logit offset, shape ``(C,)``.

    Returns:
        Packed bounds tensor of shape ``(B, 2C)``.
    """
    msg = f"No compute_efficient_credal_bounds implementation registered for array type {type(logits)}"
    raise NotImplementedError(msg)


class EfficientCredalRepresenter[**In, Out: CategoricalDistribution, C: ProbabilityIntervalsCredalSet](
    Representer[Any, In, Out, C]
):
    """Build a credal set from base logits and calibrated logit-space offsets."""

    predictor: EfficientCredalPredictor[In, Out]

    def __init__(self, predictor: EfficientCredalPredictor[In, Out]) -> None:
        """Initialize the representer with an efficient credal predictor.

        Args:
            predictor: The efficient credal prediction predictor to represent.
        """
        super().__init__(predictor)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        """Run the base, perturb each logit by the calibrated offsets, and reduce to intervals."""
        logits = predict_raw(self.predictor, *args, **kwargs)
        packed = compute_efficient_credal_bounds(logits, self.predictor.lower, self.predictor.upper)
        return create_probability_intervals_from_lower_upper_array(packed)  # ty:ignore[invalid-return-type]


representer.register(EfficientCredalPredictor, EfficientCredalRepresenter)
