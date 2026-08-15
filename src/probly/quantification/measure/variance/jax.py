"""Jax implementations of regression variance measures."""

from __future__ import annotations

from jax import numpy as jnp

from probly.representation.distribution.jax_gaussian import (
    JaxGaussianDistribution,
    JaxGaussianDistributionSample,
)
from probly.representation.sample.jax import JaxArraySample

from ._common import (
    LogBase,
    conditional_variance,
    mutual_information_variance,
    variance,
    variance_of_expected_predictive_distribution,
)


@variance.register(JaxGaussianDistribution)
def array_gaussian_variance(
    distribution: JaxGaussianDistribution | jnp.ndarray,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the variance of a Gaussian distribution."""
    if isinstance(distribution, JaxGaussianDistribution):
        return distribution.var
    return distribution


@conditional_variance.register(JaxGaussianDistributionSample)
def array_gaussian_sample_conditional_variance(
    sample: JaxGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the aleatoric variance of a Gaussian sample (mean of per-model variances)."""
    return jnp.mean(sample.array.var, axis=sample.sample_axis)


@mutual_information_variance.register(JaxGaussianDistributionSample)
def array_gaussian_sample_mutual_information(
    sample: JaxGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the epistemic variance of a Gaussian sample (variance of per-model means)."""
    return jnp.var(sample.array.mean, axis=sample.sample_axis, ddof=0)


@variance_of_expected_predictive_distribution.register(JaxGaussianDistributionSample)
def array_gaussian_sample_variance_of_expected_predictive_distribution(
    sample: JaxGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the total predictive variance of a Gaussian sample via the law of total variance."""
    return array_gaussian_sample_conditional_variance(sample) + array_gaussian_sample_mutual_information(sample)


@variance_of_expected_predictive_distribution.register(JaxArraySample)
def array_sample_variance_of_expected_predictive_distribution(
    sample: JaxArraySample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the total predictive variance of a raw jax array sample."""
    return jnp.var(sample.array, axis=sample.sample_axis, ddof=0)


@conditional_variance.register(JaxArraySample)
def array_sample_conditional_variance(
    sample: JaxArraySample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the conditional variance of a raw jax array sample (zero for point predictions)."""
    return jnp.zeros_like(jnp.mean(sample.array, axis=sample.sample_axis))


@mutual_information_variance.register(JaxArraySample)
def array_sample_mutual_information(
    sample: JaxArraySample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the epistemic variance of a raw jax array sample."""
    return jnp.var(sample.array, axis=sample.sample_axis, ddof=0)
