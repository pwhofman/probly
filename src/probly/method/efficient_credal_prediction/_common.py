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
from probly.representer._representer import Representer, representer
from probly.transformation.transformation import predictor_transformation

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


def _validate_alpha(alpha: float) -> None:
    """Validate that ``alpha`` lies in ``[0, 1]``; raise ``ValueError`` otherwise."""
    if not 0.0 <= alpha <= 1.0:
        msg = f"alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)


@flexdispatch
def compute_efficient_credal_prediction_bounds[T: ArrayLike](
    logits_train: T,
    targets_train: T,
    num_classes: int,
    alpha: float,
    **_kwargs,  # noqa: ANN003
) -> tuple[T, T]:
    """Compute per-class additive logit bounds via classwise relative-likelihood optimization.

    Dispatches to backend-specific implementations based on the array type.
    """
    msg = f"No credal bounds computation registered for array type {type(logits_train)}"
    raise NotImplementedError(msg)


@flexdispatch
def compute_efficient_credal_bounds[T: ArrayLike](logits: T, lower: T, upper: T) -> T:
    """Compute packed ``(B, 2C)`` interval probability bounds via 2K logit perturbations.

    Implements the inference step of :cite:`hofmanefficient`: for each class
    ``k``, the kth logit is perturbed by ``lower[k]`` (signed non-positive)
    and ``upper[k]`` (signed non-negative) independently of the others, and
    each perturbed logit vector is softmaxed. The packed result has the
    per-class min of the 2K resulting distributions in the first half and
    the per-class max in the second.

    Args:
        logits: Base classifier output of shape ``(B, C)``.
        lower: Per-class signed lower-direction logit offset, shape ``(C,)``.
        upper: Per-class signed upper-direction logit offset, shape ``(C,)``.

    Returns:
        Packed bounds tensor of shape ``(B, 2C)``.
    """
    msg = f"No compute_efficient_credal_bounds implementation registered for array type {type(logits)}"
    raise NotImplementedError(msg)


@representer.register(EfficientCredalPredictor)
class EfficientCredalRepresenter[**In, Out: CategoricalDistribution, C: ProbabilityIntervalsCredalSet](
    Representer[Any, In, Out, C]
):
    """Builds a credal set from the base logits and the calibrated logit-space offsets.

    For each class ``k``, the kth logit is perturbed by ``lower[k]`` and
    ``upper[k]`` (signed: ``lower`` is non-positive, ``upper`` is non-negative)
    independently of the others, and the result is softmaxed. The credal set's
    ``i``th lower (resp. upper) bound is the min (resp. max) of the ``i``th
    coordinate across the 2K resulting distributions.
    """

    predictor: EfficientCredalPredictor[In, Out]

    def __init__(self, predictor: EfficientCredalPredictor[In, Out]) -> None:
        """Initialize the representer with an efficient credal predictor."""
        super().__init__(predictor)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        """Run the base, perturb each logit by the calibrated offsets, and reduce to credal bounds."""
        if self.predictor.lower is None or self.predictor.upper is None:
            msg = (
                "EfficientCredalPredictor has uninitialized bounds; call "
                "compute_efficient_credal_prediction_bounds and assign the result to "
                "predictor.lower / predictor.upper before requesting a representation."
            )
            raise RuntimeError(msg)
        logits = predict_raw(self.predictor, *args, **kwargs)
        packed = compute_efficient_credal_bounds(logits, self.predictor.lower, self.predictor.upper)
        return create_probability_intervals_from_lower_upper_array(packed)  # ty:ignore[invalid-return-type]
