"""Tests for the point_from_mean decider."""

from __future__ import annotations

import numpy as np
import pytest

from probly.decider import point_from_mean
from probly.representation.distribution.numpy_categorical import NumpyProbabilityCategoricalDistribution
from probly.representation.distribution.numpy_gaussian import NumpyGaussianDistribution
from probly.representation.sample.numpy import NumpySample


def test_point_from_mean_passes_arrays_through() -> None:
    prediction = np.array([1.0, 2.0, 3.0])

    assert point_from_mean(prediction) is prediction


def test_point_from_mean_reduces_gaussian_to_its_mean() -> None:
    mean = np.array([0.5, -1.0])
    gaussian = NumpyGaussianDistribution(mean=mean, var=np.array([1.0, 2.0]))

    np.testing.assert_array_equal(point_from_mean(gaussian), mean)


def test_point_from_mean_reduces_sample_to_sample_mean() -> None:
    sample = NumpySample(array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), sample_axis=0)

    np.testing.assert_allclose(point_from_mean(sample), np.array([3.0, 4.0]))


def test_point_from_mean_rejects_categorical_distribution() -> None:
    distribution = NumpyProbabilityCategoricalDistribution(np.array([[0.2, 0.8]]))

    with pytest.raises(NotImplementedError, match="point_from_mean"):
        point_from_mean(distribution)
