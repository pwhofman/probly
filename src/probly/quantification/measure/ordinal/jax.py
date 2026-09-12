"""Jax implementations for ordinal classification measures."""

from __future__ import annotations

from jax import numpy as jnp

from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxCategoricalDistributionSample,
)
from probly.utils.jax import jax_entropy

from ._common import (
    LogBase,
    categorical_variance_aleatoric,
    categorical_variance_total,
    labelwise_conditional_entropy,
    labelwise_conditional_variance,
    labelwise_entropy,
    labelwise_entropy_of_expected_predictive_distribution,
    labelwise_mutual_information_entropy,
    labelwise_mutual_information_variance,
    labelwise_variance,
    labelwise_variance_of_expected_predictive_distribution,
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_mutual_information_entropy,
    ordinal_mutual_information_variance,
    ordinal_variance,
    ordinal_variance_of_expected_predictive_distribution,
)


def _jax_binary_entropy(p: jnp.ndarray, base: LogBase = None) -> jnp.ndarray:
    """Compute the binary Shannon entropy of probabilities ``p``.

    ``base="normalize"`` normalizes by ``log(2)`` so that the entropy is in
    ``[0, 1]`` for each binary problem.
    """
    p_stack = jnp.stack([p, 1 - p], axis=-1)
    entropy = jax_entropy(p_stack)
    if base is None or base == jnp.e:
        return entropy
    if base == "normalize":
        base = 2.0

    return entropy / jnp.log(jnp.array(base, dtype=entropy.dtype))


def _cdf(p: jnp.ndarray) -> jnp.ndarray:
    """Compute the cumulative distribution function (CDF) excluding the last bin."""
    return jnp.cumsum(p, axis=-1)[..., :-1]


@ordinal_variance.register(JaxCategoricalDistribution)
def jax_categorical_ordinal_variance(
    distribution: JaxCategoricalDistribution | jnp.ndarray,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the ordinal variance of a categorical distribution."""
    if isinstance(distribution, JaxCategoricalDistribution):
        p = distribution.probabilities
        del distribution
    else:
        p = distribution
    cdf = _cdf(p)
    return jnp.sum(cdf * (1 - cdf), axis=-1)


@ordinal_entropy.register(JaxCategoricalDistribution)
def jax_categorical_ordinal_entropy(
    distribution: JaxCategoricalDistribution | jnp.ndarray, base: LogBase = None
) -> jnp.ndarray:
    """Compute the ordinal entropy of a categorical distribution."""
    if isinstance(distribution, JaxCategoricalDistribution):
        p = distribution.probabilities
        del distribution
    else:
        p = distribution
    cdf = _cdf(p)
    binary_entropies = _jax_binary_entropy(cdf, base=base)
    return jnp.sum(binary_entropies, axis=-1)


@ordinal_variance_of_expected_predictive_distribution.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_ordinal_variance_of_expected_predictive_distribution(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the ordinal variance of the expected value of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    expected_cdf = jnp.mean(cdf, axis=axis)
    return jnp.sum(expected_cdf * (1 - expected_cdf), axis=-1)


@ordinal_conditional_variance.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_ordinal_conditional_variance(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the ordinal conditional variance of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    per_sample_variance = jnp.sum(cdf * (1 - cdf), axis=-1)
    return jnp.mean(per_sample_variance, axis=axis)


@ordinal_mutual_information_variance.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_ordinal_mutual_information_variance(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the ordinal mutual information (variance-based) of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    return jnp.sum(jnp.var(cdf, axis=axis, ddof=0), axis=-1)


@ordinal_entropy_of_expected_predictive_distribution.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_ordinal_entropy_of_expected_predictive_distribution(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jnp.ndarray:
    """Compute the ordinal entropy of the expected value of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    expected_cdf = jnp.mean(cdf, axis=axis)
    binary_entropies = _jax_binary_entropy(expected_cdf, base=base)
    return jnp.sum(binary_entropies, axis=-1)


@ordinal_conditional_entropy.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_ordinal_conditional_entropy(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jnp.ndarray:
    """Compute the ordinal conditional entropy of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    binary_entropies = _jax_binary_entropy(cdf, base=base)
    per_sample_entropy = jnp.sum(binary_entropies, axis=-1)
    return jnp.mean(per_sample_entropy, axis=axis)


@ordinal_mutual_information_entropy.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_ordinal_mutual_information_entropy(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jnp.ndarray:
    """Compute the ordinal mutual information (entropy-based) of a categorical sample."""
    return jax_categorical_sample_ordinal_entropy_of_expected_predictive_distribution(
        sample, base
    ) - jax_categorical_sample_ordinal_conditional_entropy(sample, base)


def _integer_labels(num_classes: int) -> jnp.ndarray:
    """Integer encoding ``1, ..., K`` as a 1-D float array."""
    return jnp.arange(1, num_classes + 1, dtype=float)


@categorical_variance_total.register(JaxCategoricalDistributionSample)
def jax_ordinal_integer_variance_total(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the total variance under integer label encoding for a jax categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1])
    expected_p = jnp.mean(p, axis=axis)
    mu = jnp.sum(labels * expected_p, axis=-1, keepdims=True)
    return jnp.sum(((labels - mu) ** 2) * expected_p, axis=-1)


@categorical_variance_aleatoric.register(JaxCategoricalDistributionSample)
def jax_ordinal_integer_variance_aleatoric(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the aleatoric variance under integer label encoding for a jax categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1])
    mu_m = jnp.sum(labels * p, axis=-1, keepdims=True)
    per_model = jnp.sum(((labels - mu_m) ** 2) * p, axis=-1)
    return jnp.mean(per_model, axis=axis)


# Label-wise (one-vs-rest) binary reduction


@labelwise_entropy.register(JaxCategoricalDistribution)
def jax_categorical_labelwise_entropy(
    distribution: JaxCategoricalDistribution | jnp.ndarray, base: LogBase = None
) -> jnp.ndarray:
    """Compute the label-wise binary entropy of a categorical distribution."""
    p = distribution.probabilities if isinstance(distribution, JaxCategoricalDistribution) else distribution
    return jnp.sum(_jax_binary_entropy(p, base=base), axis=-1)


@labelwise_variance.register(JaxCategoricalDistribution)
def jax_categorical_labelwise_variance(
    distribution: JaxCategoricalDistribution | jnp.ndarray,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the label-wise binary variance of a categorical distribution."""
    p = distribution.probabilities if isinstance(distribution, JaxCategoricalDistribution) else distribution
    return jnp.sum(p * (1 - p), axis=-1)


@labelwise_entropy_of_expected_predictive_distribution.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_labelwise_entropy_of_expected_predictive_distribution(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jnp.ndarray:
    """Compute the label-wise binary entropy of the expected predictive distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    expected_p = jnp.mean(p, axis=axis)
    return jnp.sum(_jax_binary_entropy(expected_p, base=base), axis=-1)


@labelwise_conditional_entropy.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_labelwise_conditional_entropy(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jnp.ndarray:
    """Compute the label-wise conditional entropy of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    per_sample_entropy = jnp.sum(_jax_binary_entropy(p, base=base), axis=-1)
    return jnp.mean(per_sample_entropy, axis=axis)


@labelwise_mutual_information_entropy.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_labelwise_mutual_information_entropy(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jnp.ndarray:
    """Compute the label-wise entropy-based mutual information of a categorical sample."""
    return jax_categorical_sample_labelwise_entropy_of_expected_predictive_distribution(
        sample, base
    ) - jax_categorical_sample_labelwise_conditional_entropy(sample, base)


@labelwise_variance_of_expected_predictive_distribution.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_labelwise_variance_of_expected_predictive_distribution(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the label-wise binary variance of the expected predictive distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    expected_p = jnp.mean(p, axis=axis)
    return jnp.sum(expected_p * (1 - expected_p), axis=-1)


@labelwise_conditional_variance.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_labelwise_conditional_variance(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the label-wise conditional variance of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    per_sample_variance = jnp.sum(p * (1 - p), axis=-1)
    return jnp.mean(per_sample_variance, axis=axis)


@labelwise_mutual_information_variance.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_labelwise_mutual_information_variance(
    sample: JaxCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> jnp.ndarray:
    """Compute the label-wise variance-based mutual information of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    return jnp.sum(jnp.var(p, axis=axis, ddof=0), axis=-1)
