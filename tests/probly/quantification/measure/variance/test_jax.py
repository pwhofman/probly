"""Tests for the variance measures on jax representations."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from probly.quantification.measure.variance import (
    expected_conditional_variance,
    variance_of_conditional_mean,
    variance_of_expected_predictive_distribution,
)
from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution, JaxGaussianDistributionSample
from probly.representation.sample.jax import JaxSample

# Three members for a batch of two: the member means have population variance 8/3 in both columns.
MEANS = jnp.array([[0.0, 2.0], [2.0, 4.0], [4.0, 0.0]])
VARIANCES = jnp.array([[1.0, 1.0], [2.0, 3.0], [3.0, 5.0]])


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_gaussian_sample_variance_decomposes_by_the_law_of_total_variance(sample_axis: int) -> None:
    gaussians = JaxGaussianDistribution(
        mean=jnp.moveaxis(MEANS, 0, sample_axis), var=jnp.moveaxis(VARIANCES, 0, sample_axis)
    )
    sample = JaxGaussianDistributionSample(array=gaussians, sample_axis=sample_axis)
    np.testing.assert_allclose(expected_conditional_variance(sample), [2.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(variance_of_conditional_mean(sample), [8 / 3, 8 / 3], rtol=1e-6)
    np.testing.assert_allclose(variance_of_expected_predictive_distribution(sample), [2 + 8 / 3, 3 + 8 / 3], rtol=1e-6)


def test_point_prediction_sample_variance_is_all_epistemic() -> None:
    sample = JaxSample(array=MEANS, sample_axis=0)
    np.testing.assert_allclose(variance_of_expected_predictive_distribution(sample), [8 / 3, 8 / 3], rtol=1e-6)
    np.testing.assert_allclose(variance_of_conditional_mean(sample), [8 / 3, 8 / 3], rtol=1e-6)
    np.testing.assert_array_equal(expected_conditional_variance(sample), [0.0, 0.0])
