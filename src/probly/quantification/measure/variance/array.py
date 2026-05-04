"""Numpy implementations of regression variance measures."""

from __future__ import annotations

import numpy as np

from probly.representation.distribution.array_gaussian import (
    ArrayGaussianDistribution,
    ArrayGaussianDistributionSample,
)
from probly.representation.sample.array import ArraySample

from ._common import (
    LogBase,
    conditional_variance,
    mutual_information,
    variance,
    variance_of_expected_predictive_distribution,
)


@variance.register(ArrayGaussianDistribution)
def array_gaussian_variance(
    distribution: ArrayGaussianDistribution | np.ndarray,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the variance of a Gaussian distribution."""
    if isinstance(distribution, ArrayGaussianDistribution):
        return distribution.var
    return distribution


@conditional_variance.register(ArrayGaussianDistributionSample)
def array_gaussian_sample_conditional_variance(
    sample: ArrayGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the aleatoric variance of a Gaussian sample (mean of per-model variances)."""
    return np.mean(sample.array.var, axis=sample.sample_axis)


@mutual_information.register(ArrayGaussianDistributionSample)
def array_gaussian_sample_mutual_information(
    sample: ArrayGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the epistemic variance of a Gaussian sample (variance of per-model means)."""
    return np.var(sample.array.mean, axis=sample.sample_axis, ddof=0)


@variance_of_expected_predictive_distribution.register(ArrayGaussianDistributionSample)
def array_gaussian_sample_variance_of_expected_predictive_distribution(
    sample: ArrayGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the total predictive variance of a Gaussian sample via the law of total variance."""
    return array_gaussian_sample_conditional_variance(sample) + array_gaussian_sample_mutual_information(sample)


@variance_of_expected_predictive_distribution.register(ArraySample)
def array_sample_variance_of_expected_predictive_distribution(
    sample: ArraySample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the total predictive variance of a raw numpy array sample."""
    return np.var(sample.array, axis=sample.sample_axis, ddof=0)


@conditional_variance.register(ArraySample)
def array_sample_conditional_variance(
    sample: ArraySample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the conditional variance of a raw numpy array sample (zero for point predictions)."""
    return np.zeros_like(np.mean(sample.array, axis=sample.sample_axis))


@mutual_information.register(ArraySample)
def array_sample_mutual_information(
    sample: ArraySample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the epistemic variance of a raw numpy array sample."""
    return np.var(sample.array, axis=sample.sample_axis, ddof=0)
