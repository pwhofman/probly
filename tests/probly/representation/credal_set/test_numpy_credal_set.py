"""Tests for numpy-backed categorical credal sets."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.credal_set.numpy import (
    NumpyConvexCredalSet,
    NumpyDiscreteCredalSet,
    NumpyDistanceBasedCredalSet,
    NumpyProbabilityIntervalsCredalSet,
    NumpySingletonCredalSet,
)
from probly.representation.distribution.numpy_categorical import (
    NumpyCategoricalDistribution,
    NumpyProbabilityCategoricalDistribution,
)
from probly.representation.sample.numpy import NumpySample


def test_convex_credal_set_from_distribution_sample() -> None:
    probs = np.array(
        [
            [[0.1, 0.9, 0.0], [0.2, 0.6, 0.2]],
            [[0.2, 0.8, 0.0], [0.3, 0.5, 0.2]],
            [[0.15, 0.75, 0.1], [0.25, 0.55, 0.2]],
        ],
        dtype=float,
    )
    sample = NumpySample(array=NumpyProbabilityCategoricalDistribution(probs), sample_axis=0)

    cset = NumpyConvexCredalSet.from_numpy_sample(sample)

    assert isinstance(cset.array, NumpyCategoricalDistribution)
    assert cset.array.probabilities.shape == (2, 3, 3)


def test_probability_intervals_array_and_shape_ops() -> None:
    probs = np.array(
        [
            [[0.2, 0.8], [0.5, 0.5]],
            [[0.1, 0.9], [0.4, 0.6]],
        ],
        dtype=float,
    )
    sample = NumpySample(array=NumpyProbabilityCategoricalDistribution(probs), sample_axis=0)

    cset = NumpyProbabilityIntervalsCredalSet.from_numpy_sample(sample)
    arr = np.asarray(cset)

    assert arr.shape == (2, 2, 2)

    expanded = np.expand_dims(cset, axis=0)
    assert isinstance(expanded, NumpyProbabilityIntervalsCredalSet)
    assert expanded.lower_bounds.shape == (1, 2, 2)
    assert expanded.upper_bounds.shape == (1, 2, 2)


class TestNumpyDiscreteCredalSet:
    """Discrete credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = NumpySample(
            array=NumpyProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = NumpyDiscreteCredalSet.from_numpy_sample(sample)
        assert isinstance(credal, NumpyDiscreteCredalSet)

    def test_lower_upper_barycenter(self) -> None:
        # Build a discrete credal set with two members.
        arr = NumpyProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = NumpyDiscreteCredalSet(array=arr)
        # lower / upper are min/max along the second-to-last axis.
        np.testing.assert_allclose(cred.lower(), [[0.3, 0.5]])
        np.testing.assert_allclose(cred.upper(), [[0.5, 0.7]])

    def test_num_classes(self) -> None:
        arr = NumpyProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = NumpyDiscreteCredalSet(array=arr)
        assert cred.num_classes == 2


class TestNumpyConvexCredalSet:
    """Convex credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = NumpySample(
            array=NumpyProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = NumpyConvexCredalSet.from_numpy_sample(sample)
        assert isinstance(credal, NumpyConvexCredalSet)

    def test_lower_upper(self) -> None:
        arr = NumpyProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = NumpyConvexCredalSet(array=arr)
        np.testing.assert_allclose(cred.lower(), [[0.3, 0.5]])
        np.testing.assert_allclose(cred.upper(), [[0.5, 0.7]])

    def test_num_classes(self) -> None:
        arr = NumpyProbabilityCategoricalDistribution(array=np.array([[[0.5, 0.5], [0.3, 0.7]]]))
        cred = NumpyConvexCredalSet(array=arr)
        assert cred.num_classes == 2


class TestNumpyDistanceBasedCredalSet:
    """Distance-based credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = NumpySample(
            array=NumpyProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = NumpyDistanceBasedCredalSet.from_numpy_sample(sample)
        assert isinstance(credal, NumpyDistanceBasedCredalSet)
        # Radius should equal the maximum TV distance to the mean.
        assert credal.radius.shape == (1,)

    def test_lower_upper_barycenter_with_radius(self) -> None:
        cred = NumpyDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.6]]),
            radius=np.array([0.1]),
        )
        np.testing.assert_allclose(cred.lower(), [[0.3, 0.5]])
        np.testing.assert_allclose(cred.upper(), [[0.5, 0.7]])
        # barycenter is the nominal distribution itself.
        assert isinstance(cred.barycenter, NumpyProbabilityCategoricalDistribution)

    def test_lower_clipped_at_zero(self) -> None:
        cred = NumpyDistanceBasedCredalSet(
            nominal=np.array([[0.05, 0.95]]),
            radius=np.array([0.5]),
        )
        np.testing.assert_allclose(cred.lower(), [[0.0, 0.45]])

    def test_upper_clipped_at_one(self) -> None:
        cred = NumpyDistanceBasedCredalSet(
            nominal=np.array([[0.95, 0.05]]),
            radius=np.array([0.5]),
        )
        np.testing.assert_allclose(cred.upper(), [[1.0, 0.55]])

    def test_array_dunder(self) -> None:
        cred = NumpyDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.6]]),
            radius=np.array([0.1]),
        )
        out = np.asarray(cred)
        np.testing.assert_allclose(out, [[0.4, 0.6]])

    def test_num_classes(self) -> None:
        cred = NumpyDistanceBasedCredalSet(
            nominal=np.array([[0.4, 0.6]]),
            radius=np.array([0.1]),
        )
        assert cred.num_classes == 2


class TestNumpyProbabilityIntervalsCredalSet:
    """Interval credal set behaviour."""

    def test_from_array_sample(self) -> None:
        sample = NumpySample(
            array=NumpyProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        credal = NumpyProbabilityIntervalsCredalSet.from_numpy_sample(sample)
        # Should produce intervals from the min/max of the sample probabilities.
        np.testing.assert_allclose(credal.lower_bounds, [[0.3, 0.5]])
        np.testing.assert_allclose(credal.upper_bounds, [[0.5, 0.7]])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            NumpyProbabilityIntervalsCredalSet(
                lower_bounds=np.array([[0.1, 0.2]]),
                upper_bounds=np.array([[0.5, 0.6, 0.7]]),
            )

    def test_width(self) -> None:
        cred = NumpyProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2]]),
            upper_bounds=np.array([[0.5, 0.6]]),
        )
        np.testing.assert_allclose(cred.width(), [[0.4, 0.4]])

    def test_contains(self) -> None:
        cred = NumpyProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2]]),
            upper_bounds=np.array([[0.5, 0.6]]),
        )
        # Probability inside intervals.
        assert bool(cred.contains(np.array([[0.3, 0.4]])))
        # Probability outside.
        assert not bool(cred.contains(np.array([[0.7, 0.4]])))

    def test_array_dunder(self) -> None:
        cred = NumpyProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2]]),
            upper_bounds=np.array([[0.5, 0.6]]),
        )
        arr = np.asarray(cred)
        # Stacked along axis -2: shape (1, 2, 2)
        assert arr.shape[-2:] == (2, 2)
        np.testing.assert_allclose(arr[..., 0, :], [[0.1, 0.2]])
        np.testing.assert_allclose(arr[..., 1, :], [[0.5, 0.6]])

    def test_num_classes(self) -> None:
        cred = NumpyProbabilityIntervalsCredalSet(
            lower_bounds=np.array([[0.1, 0.2, 0.3]]),
            upper_bounds=np.array([[0.4, 0.5, 0.6]]),
        )
        assert cred.num_classes == 3


class TestNumpySingletonCredalSet:
    """Singleton credal set has lower=upper=value."""

    def test_from_array_sample(self) -> None:
        sample = NumpySample(
            array=NumpyProbabilityCategoricalDistribution(
                array=np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
            ),
            sample_axis=0,
        )
        cred = NumpySingletonCredalSet.from_numpy_sample(sample)
        assert isinstance(cred, NumpySingletonCredalSet)

    def test_lower_eq_upper(self) -> None:
        arr = NumpyProbabilityCategoricalDistribution(array=np.array([[0.4, 0.6]]))
        cred = NumpySingletonCredalSet(array=arr)
        np.testing.assert_allclose(cred.lower(), [[0.4, 0.6]])
        np.testing.assert_allclose(cred.upper(), [[0.4, 0.6]])

    def test_barycenter_returns_array(self) -> None:
        arr = NumpyProbabilityCategoricalDistribution(array=np.array([[0.4, 0.6]]))
        cred = NumpySingletonCredalSet(array=arr)
        # Barycenter is the contained distribution itself.
        assert cred.barycenter is arr

    def test_num_classes(self) -> None:
        arr = NumpyProbabilityCategoricalDistribution(array=np.array([[0.4, 0.6]]))
        cred = NumpySingletonCredalSet(array=arr)
        assert cred.num_classes == 2


class TestFromSampleTypeError:
    """``from_sample`` raises TypeError when the sample's array isn't categorical."""

    def test_raises_for_non_categorical(self) -> None:
        # Plain ndarray sample, not wrapped in NumpyCategoricalDistribution.
        sample = NumpySample(array=np.array([[0.5, 0.5], [0.3, 0.7]]), sample_axis=0)
        with pytest.raises(TypeError, match="NumpyCategoricalDistribution"):
            NumpyDiscreteCredalSet.from_sample(sample)
