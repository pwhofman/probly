"""Tests for numpy-backed categorical credal sets."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)
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
    sample = ArraySample(array=ArrayProbabilityCategoricalDistribution(probs), sample_axis=0)

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
    sample = ArraySample(array=ArrayProbabilityCategoricalDistribution(probs), sample_axis=0)

    cset = ArrayProbabilityIntervalsCredalSet.from_array_sample(sample)
    arr = np.asarray(cset)

    assert arr.shape == (2, 2, 2)

    expanded = np.expand_dims(cset, axis=0)
    assert isinstance(expanded, ArrayProbabilityIntervalsCredalSet)
    assert expanded.lower_bounds.shape == (1, 2, 2)
    assert expanded.upper_bounds.shape == (1, 2, 2)


class TestArrayDiscreteCredalSet:
    """Discrete credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = ArraySample(
            array=ArrayProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = ArrayDiscreteCredalSet.from_array_sample(sample)
        assert isinstance(credal, ArrayDiscreteCredalSet)

    def test_lower_upper_barycenter(self) -> None:
        # Build a discrete credal set with two members.
        arr = ArrayProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = ArrayDiscreteCredalSet(array=arr)
        # lower / upper are min/max along the second-to-last axis.
        np.testing.assert_allclose(cred.lower(), [[0.3, 0.5]])
        np.testing.assert_allclose(cred.upper(), [[0.5, 0.7]])

    def test_num_classes(self) -> None:
        arr = ArrayProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = ArrayDiscreteCredalSet(array=arr)
        assert cred.num_classes == 2


class TestArrayConvexCredalSet:
    """Convex credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = ArraySample(
            array=ArrayProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = ArrayConvexCredalSet.from_array_sample(sample)
        assert isinstance(credal, ArrayConvexCredalSet)

    def test_lower_upper(self) -> None:
        arr = ArrayProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = ArrayConvexCredalSet(array=arr)
        np.testing.assert_allclose(cred.lower(), [[0.3, 0.5]])
        np.testing.assert_allclose(cred.upper(), [[0.5, 0.7]])

    def test_num_classes(self) -> None:
        arr = ArrayProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = ArrayConvexCredalSet(array=arr)
        assert cred.num_classes == 2


class TestArrayDistanceBasedCredalSet:
    """Distance-based credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = ArraySample(
            array=ArrayProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = ArrayDistanceBasedCredalSet.from_array_sample(sample)
        assert isinstance(credal, ArrayDistanceBasedCredalSet)
        # Radius should equal the maximum TV distance to the mean.
        assert credal.radius.shape == (1,)

    def test_lower_upper_barycenter_with_radius(self) -> None:
        cred = ArrayDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.6]]),
            radius=np.array([0.1]),
        )
        np.testing.assert_allclose(cred.lower(), [[0.3, 0.5]])
        np.testing.assert_allclose(cred.upper(), [[0.5, 0.7]])
        # barycenter is the nominal distribution itself.
        assert isinstance(cred.barycenter, ArrayProbabilityCategoricalDistribution)

    def test_lower_clipped_at_zero(self) -> None:
        cred = ArrayDistanceBasedCredalSet(
            nominal=np.array([[0.05, 0.95]]),
            radius=np.array([0.5]),
        )
        np.testing.assert_allclose(cred.lower(), [[0.0, 0.45]])

    def test_upper_clipped_at_one(self) -> None:
        cred = ArrayDistanceBasedCredalSet(
            nominal=np.array([[0.95, 0.05]]),
            radius=np.array([0.5]),
        )
        np.testing.assert_allclose(cred.upper(), [[1.0, 0.55]])

    def test_array_dunder(self) -> None:
        cred = ArrayDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.6]]),
            radius=np.array([0.1]),
        )
        out = np.asarray(cred)
        np.testing.assert_allclose(out, [[0.4, 0.6]])

    def test_num_classes(self) -> None:
        cred = ArrayDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.6]]),
            radius=np.array([0.1]),
        )
        assert cred.num_classes == 2


class TestArrayProbabilityIntervalsCredalSet:
    """Interval credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = ArraySample(
            array=ArrayProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = ArrayProbabilityIntervalsCredalSet.from_array_sample(sample)
        # Should produce intervals from the min/max of the sample probabilities.
        np.testing.assert_allclose(credal.lower_bounds, [[0.3, 0.5]])
        np.testing.assert_allclose(credal.upper_bounds, [[0.5, 0.7]])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            ArrayProbabilityIntervalsCredalSet(
                lower_bounds=np.array([[0.1, 0.2]]),
                upper_bounds=np.array([[0.5, 0.6, 0.7]]),
            )

    def test_width(self) -> None:
        cred = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2]]),
            upper_bounds=np.array([[0.5, 0.6]]),
        )
        np.testing.assert_allclose(cred.width(), [[0.4, 0.4]])

    def test_contains(self) -> None:
        cred = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2]]),
            upper_bounds=np.array([[0.5, 0.6]]),
        )
        # Probability inside intervals.
        assert bool(cred.contains(np.array([[0.3, 0.4]])))
        # Probability outside.
        assert not bool(cred.contains(np.array([[0.7, 0.4]])))

    def test_array_dunder(self) -> None:
        cred = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2]]),
            upper_bounds=np.array([[0.5, 0.6]]),
        )
        arr = np.asarray(cred)
        # Stacked along axis -2: shape (1, 2, 2)
        assert arr.shape[-2:] == (2, 2)
        np.testing.assert_allclose(arr[..., 0, :], [[0.1, 0.2]])
        np.testing.assert_allclose(arr[..., 1, :], [[0.5, 0.6]])

    def test_num_classes(self) -> None:
        cred = ArrayProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2, 0.3]]),
            upper_bounds=np.array([[0.4, 0.5, 0.6]]),
        )
        assert cred.num_classes == 3


class TestArraySingletonCredalSet:
    """Singleton credal set has lower=upper=value."""

    def test_from_array_sample(self) -> None:
        sample = ArraySample(
            array=ArrayProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        cred = ArraySingletonCredalSet.from_array_sample(sample)
        assert isinstance(cred, ArraySingletonCredalSet)

    def test_lower_eq_upper(self) -> None:
        arr = ArrayProbabilityCategoricalDistribution(array=np.array([[0.4, 0.6]]))
        cred = ArraySingletonCredalSet(array=arr)
        np.testing.assert_allclose(cred.lower(), [[0.4, 0.6]])
        np.testing.assert_allclose(cred.upper(), [[0.4, 0.6]])

    def test_barycenter_returns_array(self) -> None:
        arr = ArrayProbabilityCategoricalDistribution(array=np.array([[0.4, 0.6]]))
        cred = ArraySingletonCredalSet(array=arr)
        # Barycenter is the contained distribution itself.
        assert cred.barycenter is arr

    def test_num_classes(self) -> None:
        arr = ArrayProbabilityCategoricalDistribution(array=np.array([[0.4, 0.6]]))
        cred = ArraySingletonCredalSet(array=arr)
        assert cred.num_classes == 2


class TestFromSampleTypeError:
    """``from_sample`` raises TypeError when the sample's array isn't categorical."""

    def test_raises_for_non_categorical(self) -> None:
        # Plain ndarray sample, not wrapped in ArrayCategoricalDistribution.
        sample = ArraySample(array=np.array([[0.5, 0.5], [0.3, 0.7]]), sample_axis=0)
        with pytest.raises(TypeError, match="ArrayCategoricalDistribution"):
            ArrayDiscreteCredalSet.from_sample(sample)
