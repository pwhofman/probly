"""Entropy measures for JAX array distributions."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxCategoricalDistributionSample,
)
from probly.representation.distribution.jax_gaussian import (
    JaxGaussianDistribution,
    JaxGaussianDistributionSample,
)

from ._common import (
    LogBase,
    conditional_entropy,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    mutual_information,
)


def _jax_entropy(p: jax.Array) -> jax.Array:
    """Shannon entropy ``H(p)`` along the last axis. ``0 * log(0)`` is treated as ``0``."""
    log_p = jnp.where(p > 0, jnp.log(p), jnp.zeros_like(p))
    return -jnp.sum(p * log_p, axis=-1)


def _categorical_array(sample: JaxCategoricalDistributionSample) -> JaxCategoricalDistribution:
    """Return ``sample.array`` typed as a :class:`JaxCategoricalDistribution`.

    :class:`JaxArraySample` declares ``array: jax.Array``; subclasses specialize
    to a protected-axis distribution at runtime. The cast here is a typing-only
    refinement and never converts the value.
    """
    return cast("JaxCategoricalDistribution", sample.array)


def _gaussian_array(sample: JaxGaussianDistributionSample) -> JaxGaussianDistribution:
    """Return ``sample.array`` typed as a :class:`JaxGaussianDistribution`.

    See :func:`_categorical_array` for the typing rationale.
    """
    return cast("JaxGaussianDistribution", sample.array)


# Entropy


@entropy.register(JaxCategoricalDistribution)
def jax_categorical_entropy(distribution: JaxCategoricalDistribution | jax.Array, base: LogBase = None) -> jax.Array:
    """Compute the entropy of a categorical distribution represented as a JAX array."""
    if isinstance(distribution, JaxCategoricalDistribution):
        p = distribution.probabilities
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        p = distribution

    result = _jax_entropy(p)
    if base is None or base == jnp.e:
        return result
    if base == "normalize":
        base = float(p.shape[-1])

    return result / jnp.log(jnp.asarray(base, dtype=result.dtype))


@entropy.register(JaxGaussianDistribution)
def jax_gaussian_entropy(distribution: JaxGaussianDistribution | jax.Array, base: LogBase = None) -> jax.Array:
    """Compute the differential entropy of a Gaussian distribution represented as a JAX array.

    Takes either a ``JaxGaussianDistribution`` or a single ``jax.Array`` representing the variance.
    """
    if isinstance(distribution, JaxGaussianDistribution):
        var = distribution.var
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        var = distribution
    result = 0.5 * jnp.log(2 * jnp.e * jnp.pi * var)
    if base is None or base == jnp.e:
        return result
    if base == "normalize":
        msg = "Entropy normalization is not supported for Gaussian distributions."
        raise ValueError(msg)
    return result / jnp.log(jnp.asarray(base, dtype=result.dtype))


# Entropy of expected value


@entropy_of_expected_predictive_distribution.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_entropy_of_expected_predictive_distribution(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    p = _categorical_array(sample).probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected = jnp.mean(p, axis=axis)
    return jax_categorical_entropy(expected, base=base)


@entropy_of_expected_predictive_distribution.register(JaxGaussianDistributionSample)
def jax_gaussian_sample_entropy_of_expected_predictive_distribution(
    sample: JaxGaussianDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the entropy of the expected Gaussian via the law of total variance."""
    axis = sample.sample_axis
    array = _gaussian_array(sample)
    del sample  # Avoid keeping a reference to the sample for memory efficiency

    # We compute the entropy of the moment-matched Gaussian as an approximation.
    # This is an overestimate of the true entropy of the expected value,
    # which would require computing the entropy of a Gaussian mixture.
    # Interpreting this value as total uncertainty, this means that epistemic uncertainty
    # may be overestimated as-well, while aleatoric uncertainty is computed correctly.
    var = jnp.mean(array.var, axis=axis) + jnp.var(array.mean, axis=axis)
    return jax_gaussian_entropy(var, base=base)


# Conditional entropy


@conditional_entropy.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_conditional_entropy(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the conditional entropy of a sample from a categorical distribution."""
    p = _categorical_array(sample).probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    entropies = jax_categorical_entropy(p, base=base)
    return jnp.mean(entropies, axis=axis)


@conditional_entropy.register(JaxGaussianDistributionSample)
def jax_gaussian_sample_conditional_entropy(sample: JaxGaussianDistributionSample, base: LogBase = None) -> jax.Array:
    """Compute the mean per-tree Gaussian entropy (aleatoric uncertainty)."""
    axis = sample.sample_axis
    entropies = jax_gaussian_entropy(_gaussian_array(sample), base=base)
    return jnp.mean(entropies, axis=axis)


# Mutual information


@mutual_information.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_mutual_information(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the mutual information of a sample from a categorical distribution."""
    p = _categorical_array(sample).probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value_entropy = jax_categorical_entropy(jnp.mean(p, axis=axis), base=base)
    conditional_entropy_value = jnp.mean(jax_categorical_entropy(p, base=base), axis=axis)
    return expected_value_entropy - conditional_entropy_value


@mutual_information.register(JaxGaussianDistributionSample)
def jax_gaussian_sample_mutual_information(sample: JaxGaussianDistributionSample, base: LogBase = None) -> jax.Array:
    """Compute the epistemic uncertainty (total entropy minus aleatoric entropy)."""
    return jax_gaussian_sample_entropy_of_expected_predictive_distribution(
        sample, base=base
    ) - jax_gaussian_sample_conditional_entropy(sample, base=base)


# Zero-one proper scoring rule measures


@max_probability_complement_of_expected.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_max_probability_complement_of_expected(
    sample: JaxCategoricalDistributionSample,
) -> jax.Array:
    """Compute one minus the max probability of the expected value of a categorical sample."""
    p = _categorical_array(sample).probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected = jnp.mean(p, axis=axis)
    return 1.0 - jnp.max(expected, axis=-1)


@expected_max_probability_complement.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_expected_max_probability_complement(
    sample: JaxCategoricalDistributionSample,
) -> jax.Array:
    """Compute the expected value of one minus the max probability of a categorical sample."""
    p = _categorical_array(sample).probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    per_sample_complement = 1.0 - jnp.max(p, axis=-1)
    return jnp.mean(per_sample_complement, axis=axis)


@max_disagreement.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_max_disagreement(
    sample: JaxCategoricalDistributionSample,
) -> jax.Array:
    """Compute the expected gap between each sample's max probability and its probability on the BMA argmax."""
    p = _categorical_array(sample).probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = jnp.mean(p, axis=axis, keepdims=True)
    bma_argmax = jnp.argmax(expected_value, axis=-1, keepdims=True)
    per_sample_bma_prob = jnp.squeeze(jnp.take_along_axis(p, bma_argmax, axis=-1), axis=-1)
    per_sample_max = jnp.max(p, axis=-1)
    return jnp.mean(per_sample_max - per_sample_bma_prob, axis=axis)
