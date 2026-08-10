"""Entropy measures for jax array distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import numpy as np

from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxCategoricalDistributionSample,
)
from probly.representation.jax_functions import jax_mean, jax_moveaxis, jax_take_along_axis
from probly.utils.jax import jax_entropy

from ._common import (
    TOTAL_VARIATION_BISECTION_ITERATIONS,
    LogBase,
    conditional_entropy,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_generalized_entropy,
    expected_max_probability_complement,
    generalized_entropy_of_expected,
    max_disagreement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
    mutual_information,
)

if TYPE_CHECKING:
    from probly.quantification.scoring_rule import ScoringRule

# Entropy


@entropy.register
def jax_categorical_entropy(distribution: JaxCategoricalDistribution | jax.Array, base: LogBase = None) -> jax.Array:
    """Compute the entropy of a categorical distribution represented as a jax array."""
    if isinstance(distribution, JaxCategoricalDistribution):
        p = distribution.probabilities
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        p = distribution

    result = jax_entropy(p)

    if base is None or base == jnp.e:
        return result
    if base == "normalize":
        base = p.shape[-1]

    return result / jnp.log(jnp.asarray(base, dtype=result.dtype))


# Entropy of expected value


@entropy_of_expected_predictive_distribution.register(JaxCategoricalDistributionSample)
def jax_categoircal_sample_entropy_of_expected_predictive_distribution(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    expected_distribution = sample.sample_mean()
    return jax_categorical_entropy(expected_distribution, base=base)


# Conditional entropy


@conditional_entropy.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_conditional_entropy(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the conditional entropy of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    entropies = jax_categorical_entropy(p, base=base)
    return jax_mean(entropies, axis=axis)


# Mutual information


@mutual_information.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_mutual_information(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the mutual information of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value_entropy = jax_categorical_entropy(jax_mean(p, axis=axis), base=base)
    conditional_entropy_value = jax_mean(jax_categorical_entropy(p, base=base), axis=axis)
    return expected_value_entropy - conditional_entropy_value


# Zero-one proper scoring rule measures


@max_probability_complement_of_expected.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_max_probability_complement_of_expected(
    sample: JaxCategoricalDistributionSample,
) -> jax.Array:
    """Compute one minus the max probability of the expected value of a categorical sample."""
    expected_distribution = sample.sample_mean()
    return 1.0 - jnp.max(expected_distribution.probabilities, axis=-1)


@expected_max_probability_complement.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_expected_max_probability_complement(
    sample: JaxCategoricalDistributionSample,
) -> jax.Array:
    """Compute the expected value of one minus the max probability of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    per_sample_complement = 1.0 - jnp.max(p, axis=-1)
    return jax_mean(per_sample_complement, axis=axis)


@max_disagreement.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_max_disagreement(
    sample: JaxCategoricalDistributionSample,
) -> jax.Array:
    """Compute the expected gap between each sample's max probability and its probability on the BMA argmax."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = jax_mean(p, axis=axis, keepdims=True)
    bma_argmax = jnp.argmax(expected_value, axis=-1, keepdims=True)
    per_sample_bma_prob = jax_take_along_axis(p, bma_argmax, axis=-1).squeeze(-1)
    per_sample_max = jnp.max(p, axis=-1)
    return jax_mean(per_sample_max - per_sample_bma_prob, axis=axis)


# Generalized-entropy (scoring rule) measures


@generalized_entropy_of_expected.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_generalized_entropy_of_expected(
    sample: JaxCategoricalDistributionSample, scoring_rule: ScoringRule
) -> jax.Array:
    """Compute G(thetha_bar) = <theta_bar, loss(theta_bar)> for a categorical sample."""
    mean = sample.sample_mean().probabilities  # (..., K)
    # 0 * inf = 0: a zero-probability outcome contributes nothing to the expected loss.
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = mean * scoring_rule.loss(mean)
    return jnp.where(mean > 0, weighted, 0.0).sum(axis=-1)


@expected_generalized_entropy.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_expected_generalized_entropy(
    sample: JaxCategoricalDistributionSample, scoring_rule: ScoringRule
) -> jax.Array:
    """Compute E[G(theta)] = mean_m <theta_m, loss(theta_m)> for a categorical sample."""
    p = sample.array.probabilities  # (..., M, K)
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    # 0 * inf = 0: a zero-probability outcome contributes nothing to the expected loss.
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = p * scoring_rule.loss(p)
    per_sample = jnp.where(p > 0, weighted, 0.0).sum(axis=-1)  # (..., M)
    return jax_mean(per_sample, axis=axis)


# Distance-based epistemic uncertainty (Wasserstein)


def _min_expected_total_variation_from_samples(probabilities: jax.Array, sample_axis: int) -> jax.Array:
    """Solve ``1/2 min_q E_s ||p_s - q||_1`` over the simplex for a sample of distributions.

    Each ``q_k`` is the ``(1/2 - lambda)`` quantile of the marginal draws where ``lambda is the
    single multiplier that makes ``q`` sum to one. The simplex sum is monotone in the quantile
    level, so ``lambda`` is found by bisection.
    """
    probabilities = jax_moveaxis(probabilities, sample_axis, -2)  # (..., num_samples, num_classes)
    num_samples = probabilities.shape[-2]
    num_classes = probabilities.shape[-1]
    batch_shape = probabilities.shape[:-2]
    sorted_probabilities = jnp.sort(probabilities, axis=-2)

    def quantile_at(level: jax.Array) -> jax.Array:
        position = level * (num_samples - 1)  # (...)
        lower = jnp.floor(position).astype(jnp.int32)
        upper = jnp.minimum(lower + 1, num_samples - 1)
        fraction = (position - lower)[..., None]  # (..., 1)
        lower_index = jnp.broadcast_to(lower[..., None, None], (*batch_shape, 1, num_classes))
        upper_index = jnp.broadcast_to(upper[..., None, None], (*batch_shape, 1, num_classes))
        value_lower = jax_take_along_axis(sorted_probabilities, lower_index, axis=-2)[..., 0, :]  # (..., num_classes)
        value_upper = jax_take_along_axis(sorted_probabilities, upper_index, axis=-2)[..., 0, :]
        return value_lower + fraction * (value_upper - value_lower)

    # sum_k q_k(level) increses with the quantile level, so bisect for sum == 1.
    low = jnp.zeros(batch_shape)
    high = jnp.ones(batch_shape)
    for _ in range(TOTAL_VARIATION_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        below_target = quantile_at(mid).sum(axis=-1) < 1.0
        low = jnp.where(below_target, mid, low)
        high = jnp.where(below_target, high, mid)
    optimal_q = quantile_at(0.5 * (low + high))  # (..., num_classes)
    distances = jnp.abs(probabilities - optimal_q[..., None, :])  # (..., num_samples, num_classes)
    return 0.5 * jax_mean(distances, axis=-2).sum(axis=-1)


@min_expected_total_variation.register(JaxCategoricalDistributionSample)
def jax_categorical_sample_min_expected_total_variation(
    sample: JaxCategoricalDistributionSample,
) -> jax.Array:
    """Compute the distance-based epistemic uncertainty of a categorical sample."""
    probabilities = sample.array.probabilities
    sample_axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    return _min_expected_total_variation_from_samples(probabilities, sample_axis)
