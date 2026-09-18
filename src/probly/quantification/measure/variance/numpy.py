"""Numpy implementations of regression variance measures."""

from __future__ import annotations

import numpy as np

from probly.representation.distribution.numpy_gaussian import (
    NumpyGaussianDistribution,
    NumpyGaussianDistributionSample,
)
from probly.representation.sample.numpy import NumpySample

from ._common import (
    LogBase,
    conditional_variance,
    mutual_information_variance,
    variance,
    variance_of_expected_predictive_distribution,
)


@variance.register(NumpyGaussianDistribution)
def numpy_gaussian_variance(
    distribution: NumpyGaussianDistribution | np.ndarray,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the variance of a Gaussian distribution."""
    if isinstance(distribution, NumpyGaussianDistribution):
        return distribution.var
    return distribution


@conditional_variance.register(NumpyGaussianDistributionSample)
def numpy_gaussian_sample_conditional_variance(
    sample: NumpyGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the aleatoric variance of a Gaussian sample (mean of per-model variances)."""
    return np.mean(sample.array.var, axis=sample.sample_axis)


@mutual_information_variance.register(NumpyGaussianDistributionSample)
def numpy_gaussian_sample_mutual_information(
    sample: NumpyGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the epistemic variance of a Gaussian sample (variance of per-model means)."""
    return np.var(sample.array.mean, axis=sample.sample_axis, ddof=0)


@variance_of_expected_predictive_distribution.register(NumpyGaussianDistributionSample)
def numpy_gaussian_sample_variance_of_expected_predictive_distribution(
    sample: NumpyGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the total predictive variance of a Gaussian sample via the law of total variance."""
    return numpy_gaussian_sample_conditional_variance(sample) + numpy_gaussian_sample_mutual_information(sample)


@variance_of_expected_predictive_distribution.register(NumpySample)
def numpy_sample_variance_of_expected_predictive_distribution(
    sample: NumpySample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the total predictive variance of a raw numpy array sample."""
    return np.var(sample.array, axis=sample.sample_axis, ddof=0)


@conditional_variance.register(NumpySample)
def numpy_sample_conditional_variance(
    sample: NumpySample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the conditional variance of a raw numpy array sample (zero for point predictions)."""
    return np.zeros_like(np.mean(sample.array, axis=sample.sample_axis))


@mutual_information_variance.register(NumpySample)
def numpy_sample_mutual_information(
    sample: NumpySample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the epistemic variance of a raw numpy array sample."""
    return np.var(sample.array, axis=sample.sample_axis, ddof=0)
