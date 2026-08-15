"""Entropy measures for jax array distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
import numpy as np
from scipy import special

from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxCategoricalDistributionSample,
)
from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution
from probly.representation.distribution.jax_gaussian import (
    JaxGaussianDistribution,
    JaxGaussianDistributionSample,
)
from probly.representation.jax_functions import (
    jax_mean,
    jax_moveaxis,
    jax_squeeze,
    jax_sum,
    jax_take_along_axis,
    jax_var,
)
from probly.utils.jax import jax_entropy

from ._common import (
    DEFAULT_MEAN_FIELD_FACTOR,
    DEFAULT_NUM_SAMPLES,
    TOTAL_VARIATION_BISECTION_ITERATIONS,
    LogBase,
    conditional_entropy,
    dempster_shafer_uncertainty,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_generalized_entropy,
    expected_max_probability_complement,
    generalized_entropy_of_expected,
    max_disagreement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
    mutual_information,
    vacuity,
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


@entropy.register(JaxDirichletDistribution)
def jax_dirichlet_entropy(distribution: JaxDirichletDistribution | jax.Array, base: LogBase = None) -> jax.Array:
    """Compute the (differential) entropy of a Dirichlet distribution represented as a jax array."""
    if isinstance(distribution, JaxDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = jax_sum(alphas, axis=-1)
    K = alphas.shape[-1]  # noqa: N806

    log_beta = jax_sum(special.gammaln(alphas), axis=-1) - special.gammaln(alpha_0)
    digamma_sum = (alpha_0 - K) * special.digamma(alpha_0)
    digamma_individual = jax_sum((alphas - 1) * special.digamma(alphas), axis=-1)

    res = log_beta + digamma_sum - digamma_individual

    if base is None or base == jnp.e:
        return res
    if base == "normalize":
        msg = "Entropy normalization is not supported for Dirichlet distributions."
        raise ValueError(msg)

    return res / jnp.log(base)


@entropy.register(JaxGaussianDistribution)
def jax_gaussian_entropy(distribution: JaxGaussianDistribution | jax.Array, base: LogBase = None) -> jax.Array:
    """Compute the (differential) entropy of a Gaussian distribution represented as a jax array.

    Takes either a `JaxGaussianDistribution" or a single jax.Array representing the variance.
    """
    if isinstance(distribution, JaxGaussianDistribution):
        var = distribution.var
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        var = distribution
    entropy = 0.5 * jnp.log(2 * jnp.e * jnp.pi * var)
    if base is None or base == jnp.e:
        return entropy
    if base == "normalize":
        msg = "Entropy normalization is not supported for Gaussian distributions."
        raise ValueError(msg)
    return entropy / jnp.log(base)


# Entropy of expected value


@entropy_of_expected_predictive_distribution.register(JaxDirichletDistribution)
def jax_dirichlet_entropy_of_expected_predictive_distribution(
    distribution: JaxDirichletDistribution | jax.Array, base: LogBase = None
) -> jax.Array:
    """Compute the entropy of the expected value of a Dirichlet distribution."""
    if isinstance(distribution, jax.Array):
        distribution = JaxDirichletDistribution(alphas=distribution)

    expected_distribution = distribution.mean
    return jax_categorical_entropy(expected_distribution, base=base)


@entropy_of_expected_predictive_distribution.register(JaxGaussianDistributionSample)
def jax_gaussian_sample_entropy_of_expected_predictive_distribution(
    sample: JaxGaussianDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the entropy of the expectedGaussian via the law of total variance."""
    axis = sample.sample_axis
    array = sample.array
    del sample  # Avoid keeping a reference to the distribution for memory efficiency

    # We compute the entropy of the moment-matched Gaussian as an approximation.
    # This is an overestimate of the true entropy of the expected value,
    # which would require computing the entropy of a Gaussian mixture.
    # Interpreting this value as total uncertainty, this means that epistemic uncertainty
    # may be overestimated as-well, while aleatoric ucnertainty is computed correctly.
    var = jax_mean(array.var, axis=axis) + jax_var(array.mean, axis=axis)
    return jax_gaussian_entropy(var, base=base)


@entropy_of_expected_predictive_distribution.register(JaxCategoricalDistributionSample)
def jax_categoircal_sample_entropy_of_expected_predictive_distribution(
    sample: JaxCategoricalDistributionSample, base: LogBase = None
) -> jax.Array:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    expected_distribution = sample.sample_mean()
    return jax_categorical_entropy(expected_distribution, base=base)


# Conditional entropy


@conditional_entropy.register(JaxDirichletDistribution)
def jax_dirichlet_conditional_entropy(
    distribution: JaxDirichletDistribution | jax.Array, base: LogBase = None
) -> jax.Array:
    """Compute the conditional entropy of a Dirichlet distribution."""
    if isinstance(distribution, JaxDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = jax_sum(alphas, axis=-1, keepdims=True)
    mean = alphas / alpha_0

    res = jax_squeeze(special.digamma(alpha_0 + 1.0), axis=-1) - jax_sum(mean * special.digamma(alphas + 1.0), axis=-1)

    if base is None or base == jnp.e:
        return res
    if base == "normalize":
        msg = "Entropy normalization is not supported for Dirchlet distributions."
        raise ValueError(msg)

    return res / jnp.log(base)


@conditional_entropy.register(JaxGaussianDistributionSample)
def jax_gaussian_sample_conditional_entropy(sample: JaxGaussianDistributionSample, base: LogBase = None) -> jax.Array:
    """Compute the mean per-tree Gaussian entropy (aleatoric uncertainty)."""
    axis = sample.sample_axis
    entropies = jax_gaussian_entropy(sample.array, base=base)
    return jax_mean(entropies, axis=axis)


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


@mutual_information.register(JaxDirichletDistribution)
def jax_dirichlet_mutual_information(
    distribution: JaxDirichletDistribution | jax.Array, base: LogBase = None
) -> jax.Array:
    """Compute the mutual information of a Dirichlet distribution."""
    if isinstance(distribution, JaxDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    return jax_dirichlet_entropy_of_expected_predictive_distribution(
        alphas, base=base
    ) - jax_dirichlet_conditional_entropy(alphas, base=base)


@mutual_information.register(JaxGaussianDistributionSample)
def jax_gaussian_sample_mutual_information(sample: JaxGaussianDistributionSample, base: LogBase = None) -> jax.Array:
    """Compute the epistemic uncertainty (total entropy minus aleatoric entropy)."""
    return jax_gaussian_sample_entropy_of_expected_predictive_distribution(
        sample, base=base
    ) - jax_gaussian_sample_conditional_entropy(sample, base=base)


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


@expected_max_probability_complement.register(JaxDirichletDistribution)
def jax_dirichlet_expected_max_probability_complement(
    distribution: JaxDirichletDistribution,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    generator: jax.Array | None = None,
) -> jax.Array:
    """Estimate ``1 - E[max_k p_k]`` for a Dirchlet by Monte-Carlo (no closed form)."""
    if generator is None:
        generator = jax.random.key(1)
    sample = distribution.sample(num_samples, prng_key=generator)
    return jax_categorical_sample_expected_max_probability_complement(sample)


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


@min_expected_total_variation.register(JaxDirichletDistribution)
def jax_dirichlet_min_expected_total_variation(
    distribution: JaxDirichletDistribution,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    generator: jax.Array | None = None,
) -> jax.Array:
    """Estimate the distance-based epistemic uncertainty of a Dirichlet by Monte-Carlo."""
    if generator is None:
        generator = jax.random.key(1)
    sample = distribution.sample(num_samples, prng_key=generator)
    return jax_categorical_sample_min_expected_total_variation(sample)


# Vacuity


@vacuity.register(JaxDirichletDistribution)
def jax_dirichlet_vacuity(distribution: JaxDirichletDistribution | jax.Array) -> jax.Array:
    """Compute the vacuity K / alpha_0 of a Dirichlet distribution."""
    if isinstance(distribution, JaxDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the sample for memory efficiency
    else:
        alphas = distribution

    num_classes = alphas.shape[-1]
    alpha_0 = jax_sum(alphas, axis=-1)
    return jnp.asarray(num_classes / alpha_0)


@max_probability_complement_of_expected.register(JaxDirichletDistribution)
def jax_dirichlet_max_probability_complement_of_expected(
    distribution: JaxDirichletDistribution | jax.Array,
) -> jax.Array:
    """Compute ones minus the max probability of the mean of a Dirichlet distribution.

    Closed form: ``1 - max_c (alpha_c / alpha_0)``.
    """
    if isinstance(distribution, JaxDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the sample for memory efficiency
    else:
        alphas = distribution

    alpha_0 = jax_sum(alphas, axis=-1, keepdims=True)
    mean = alphas / alpha_0
    return 1.0 - jnp.max(mean, axis=-1)


# Dempster-Shafer uncertainty


@dempster_shafer_uncertainty.register(JaxGaussianDistribution)
def jax_gaussian_dempster_shafer_uncertainty(
    distribution: JaxGaussianDistribution,
    mean_field_factor: float = DEFAULT_MEAN_FIELD_FACTOR,
) -> jax.Array:
    """Compute the Dempster-Shafer unceratinty of a Gaussian over logits."""
    mean = distribution.mean
    var = distribution.var
    del distribution  # Avoid keeping a reference to the sample for memory efficiency

    num_classes = mean.shape[-1]
    adjusted = mean / jnp.sqrt(1.0 + mean_field_factor * var)
    return num_classes / (num_classes + jax_sum(jnp.exp(adjusted), axis=-1))
