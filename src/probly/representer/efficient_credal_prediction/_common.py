"""Shared representer for the efficient credal prediction method."""

from __future__ import annotations

from typing import Any, override

from flextype import flexdispatch
from probly.method.efficient_credal_prediction import EfficientCredalPredictor
from probly.predictor import predict_raw
from probly.representation.array_like import ArrayLike
from probly.representation.credal_set import (
    ProbabilityIntervalsCredalSet,
    create_probability_intervals_from_lower_upper_array,
)
from probly.representation.distribution import CategoricalDistribution
from probly.representer._representer import Representer, representer


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
        logits = predict_raw(self.predictor, *args, **kwargs)
        packed = compute_efficient_credal_bounds(logits, self.predictor.lower, self.predictor.upper)
        return create_probability_intervals_from_lower_upper_array(packed)  # ty:ignore[invalid-return-type]
