"""Tests for the variance measures on numpy representations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.measure.variance import (
    expected_conditional_variance,
    variance,
    variance_of_conditional_mean,
    variance_of_expected_predictive_distribution,
)
from probly.representation.distribution.numpy_gaussian import NumpyGaussianDistribution, NumpyGaussianDistributionSample
from probly.representation.sample.numpy import NumpySample

# Three members for a batch of two: the member means have population variance 8/3 in both columns.
MEANS = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 0.0]])
VARIANCES = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 5.0]])


def test_gaussian_variance_is_its_variance_parameter() -> None:
    distribution = NumpyGaussianDistribution(mean=MEANS, var=VARIANCES)
    np.testing.assert_array_equal(variance(distribution), VARIANCES)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_gaussian_sample_variance_decomposes_by_the_law_of_total_variance(sample_axis: int) -> None:
    gaussians = NumpyGaussianDistribution(
        mean=np.moveaxis(MEANS, 0, sample_axis), var=np.moveaxis(VARIANCES, 0, sample_axis)
    )
    sample = NumpyGaussianDistributionSample(array=gaussians, sample_axis=sample_axis)
    np.testing.assert_allclose(expected_conditional_variance(sample), [2.0, 3.0])
    np.testing.assert_allclose(variance_of_conditional_mean(sample), [8 / 3, 8 / 3])
    np.testing.assert_allclose(variance_of_expected_predictive_distribution(sample), [2 + 8 / 3, 3 + 8 / 3])


def test_point_prediction_sample_variance_is_all_epistemic() -> None:
    sample = NumpySample(array=MEANS, sample_axis=0)
    np.testing.assert_allclose(variance_of_expected_predictive_distribution(sample), [8 / 3, 8 / 3])
    np.testing.assert_allclose(variance_of_conditional_mean(sample), [8 / 3, 8 / 3])
    np.testing.assert_array_equal(expected_conditional_variance(sample), [0.0, 0.0])
