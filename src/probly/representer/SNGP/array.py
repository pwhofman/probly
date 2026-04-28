"""Array specific functionality for the SNGP representer."""

from __future__ import annotations

from typing import Any, cast

from probly.representation.distribution._common import create_categorical_distribution_from_logits
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.sample.array import ArraySample
from probly.representer.sngp._common import compute_categorical_sample_from_logits


@compute_categorical_sample_from_logits.register(ArraySample)
def array_compute_categorical_sample_from_logits(
    sample: ArraySample[Any],
) -> ArrayCategoricalDistributionSample:
    """Convert an ArraySample of SNGP logits to a categorical distribution sample."""
    array = sample.array
    sample_axis = sample.sample_axis
    if array.ndim >= 3 and sample_axis == 0:
        array = array.swapaxes(0, 1)
        sample_axis = 1

    categorical_dist = cast("ArrayCategoricalDistribution", create_categorical_distribution_from_logits(array))
    return ArrayCategoricalDistributionSample(array=categorical_dist, sample_axis=sample_axis)
