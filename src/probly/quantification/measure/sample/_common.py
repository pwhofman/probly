"""Common definitions of sample-based uncertainty measures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch
from probly.quantification._quantification import measure_atomic
from probly.representation.distribution._common import CategoricalDistributionSample
from probly.representation.sample._common import Sample

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike


@measure_atomic.register(Sample)
def sample_variance[T](sample: Sample[T]) -> T:
    """Measure uncertainty for samples via their sample variance."""
    return sample.sample_var()


@flexdispatch
def mean_squared_distance_to_scaled_one_hot(
    sample: CategoricalDistributionSample, scale: float | None = None
) -> ArrayLike:
    r"""Mean over members of :math:`\| h_k - s\, e_{\arg\max_j h_{k,j}} \|_2^2`; class axis is last.

    Args:
        sample: Logits sample across ensemble members.
        scale: Scale ``s``; ``None`` uses K (num classes). Matches DARE Eq. 35
            :cite:`mathelinDeepAntiregularizedEnsembles2023`.
    """
    msg = f"mean_squared_distance_to_scaled_one_hot not supported for sample of type {type(sample)}."
    raise NotImplementedError(msg)


@measure_atomic.register(CategoricalDistributionSample)
@flexdispatch
def total_logit_sample_variance(sample: CategoricalDistributionSample) -> ArrayLike:
    """Measure uncertainty for samples via the variance of their total logits (logits summed across members)."""
    msg = f"total_logit_sample_variance not supported for sample of type {type(sample)}."
    raise NotImplementedError(msg)
