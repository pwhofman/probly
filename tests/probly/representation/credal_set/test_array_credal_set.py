"""Tests for numpy-backed categorical credal sets."""

from __future__ import annotations

import numpy as np

from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.sample.array import ArraySample


def test_convex_credal_set_from_distribution_sample() -> None:
    probs = np.array(
        [
            [[0.1, 0.9, 0.0], [0.2, 0.6, 0.2]],
            [[0.2, 0.8, 0.0], [0.3, 0.5, 0.2]],
            [[0.15, 0.75, 0.1], [0.25, 0.55, 0.2]],
        ],
        dtype=float,
    )
    sample = ArraySample(array=ArrayCategoricalDistribution(probabilities=probs), sample_axis=0)

    cset = ArrayConvexCredalSet.from_array_sample(sample)

    assert isinstance(cset.array, ArrayCategoricalDistribution)
    assert cset.array.probabilities.shape == (2, 3, 3)


def test_probability_intervals_array_and_shape_ops() -> None:
    probs = np.array(
        [
            [[0.2, 0.8], [0.5, 0.5]],
            [[0.1, 0.9], [0.4, 0.6]],
        ],
        dtype=float,
    )
    sample = ArraySample(array=ArrayCategoricalDistribution(probabilities=probs), sample_axis=0)

    cset = ArrayProbabilityIntervalsCredalSet.from_array_sample(sample)
    arr = np.asarray(cset)

    assert arr.shape == (2, 2, 2)

    expanded = np.expand_dims(cset, axis=0)
    assert isinstance(expanded, ArrayProbabilityIntervalsCredalSet)
    assert expanded.lower_bounds.shape == (1, 2, 2)
    assert expanded.upper_bounds.shape == (1, 2, 2)
