"""Entropy measures for numpy array distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import special
from scipy.stats import entropy as scipy_entropy

from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution
from probly.representation.distribution.array_gaussian import (
    ArrayGaussianDistribution,
    ArrayGaussianDistributionSample,
)

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
    from probly.quantification.scoring_rule import ProperScoringRule

# Entropy


@entropy.register
def array_categorical_entropy(
    distribution: ArrayCategoricalDistribution | np.ndarray, base: LogBase = None
) -> np.ndarray:
    """Compute the entropy of a categorical distribution represented as a numpy array."""
    if isinstance(distribution, ArrayCategoricalDistribution):
        p = distribution.probabilities
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        p = distribution

    if base == "normalize":
        base = p.shape[-1]

    return scipy_entropy(p, axis=-1, base=base)


@entropy.register(ArrayDirichletDistribution)
def array_dirichlet_entropy(distribution: ArrayDirichletDistribution | np.ndarray, base: LogBase = None) -> np.ndarray:
    """Compute the (differential) entropy of a Dirichlet distribution represented as a numpy array."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = np.sum(alphas, axis=-1)
    K = alphas.shape[-1]  # noqa: N806

    log_beta = np.sum(special.gammaln(alphas), axis=-1) - special.gammaln(alpha_0)
    digamma_sum = (alpha_0 - K) * special.digamma(alpha_0)
    digamma_individual = np.sum((alphas - 1) * special.digamma(alphas), axis=-1)

    res = log_beta + digamma_sum - digamma_individual

    if base is None or base == np.e:
        return res
    if base == "normalize":
        msg = "Entropy normalization is not supported for Dirichlet distributions."
        raise ValueError(msg)

    return res / np.log(base)


@entropy.register(ArrayGaussianDistribution)
def array_gaussian_entropy(distribution: ArrayGaussianDistribution | np.ndarray, base: LogBase = None) -> np.ndarray:
    """Compute the (differential) entropy of a Gaussian distribution represented as a numpy array.

    Takes either an `ArrayGaussianDistribution` or a single np.ndarray representing the variance.
    """
    if isinstance(distribution, ArrayGaussianDistribution):
        var = distribution.var
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        var = distribution
    entropy = 0.5 * np.log(2 * np.e * np.pi * var)
    if base is None or base == np.e:
        return entropy
    if base == "normalize":
        msg = "Entropy normalization is not supported for Gaussian distributions."
        raise ValueError(msg)
    return entropy / np.log(base)


# Entropy of expected value


@entropy_of_expected_predictive_distribution.register(ArrayDirichletDistribution)
def array_dirichlet_entropy_of_expected_predictive_distribution(
    distribution: ArrayDirichletDistribution | np.ndarray, base: LogBase = None
) -> np.ndarray:
    """Compute the entropy of the expected value of a Dirichlet distribution."""
    if isinstance(distribution, np.ndarray):
        distribution = ArrayDirichletDistribution(alphas=distribution)

    expected_distribution = distribution.mean
    return array_categorical_entropy(expected_distribution, base=base)


@entropy_of_expected_predictive_distribution.register(ArrayGaussianDistributionSample)
def array_gaussian_sample_entropy_of_expected_predictive_distribution(
    sample: ArrayGaussianDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the entropy of the expected Gaussian via the law of total variance."""
    axis = sample.sample_axis
    array = sample.array
    del sample  # Avoid keeping a reference to the sample for memory efficiency

    # We compute the entropy of the moment-matched Gaussian as an approximation.
    # This is an overestimate of the true entropy of the expected value,
    # which would require computing the entropy of a Gaussian mixture.
    # Interpreting this value as total uncertainty, this means that epistemic uncertainty
    # may be overestimated as-well, while aleatoric uncertainty is computed correctly.
    var = np.mean(array.var, axis=axis) + np.var(array.mean, axis=axis)
    return array_gaussian_entropy(var, base=base)


@entropy_of_expected_predictive_distribution.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_entropy_of_expected_predictive_distribution(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    expected_distribution = sample.sample_mean()
    return array_categorical_entropy(expected_distribution, base=base)


# Conditional entropy


@conditional_entropy.register(ArrayDirichletDistribution)
def array_dirichlet_conditional_entropy(
    distribution: ArrayDirichletDistribution | np.ndarray, base: LogBase = None
) -> np.ndarray:
    """Compute the conditional entropy of a Dirichlet distribution."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = np.sum(alphas, axis=-1, keepdims=True)
    mean = alphas / alpha_0

    res = special.digamma(alpha_0 + 1.0).squeeze(-1) - np.sum(mean * special.digamma(alphas + 1.0), axis=-1)

    if base is None or base == np.e:
        return res
    if base == "normalize":
        msg = "Entropy normalization is not supported for Dirichlet distributions."
        raise ValueError(msg)

    return res / np.log(base)


@conditional_entropy.register(ArrayGaussianDistributionSample)
def array_gaussian_sample_conditional_entropy(
    sample: ArrayGaussianDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the mean per-tree Gaussian entropy (aleatoric uncertainty)."""
    axis = sample.sample_axis
    entropies = array_gaussian_entropy(sample.array, base=base)
    return np.mean(entropies, axis=axis)


@conditional_entropy.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_conditional_entropy(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the conditional entropy of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    entropies = array_categorical_entropy(p, base=base)
    return np.mean(entropies, axis=axis)


# Mutual information


@mutual_information.register(ArrayDirichletDistribution)
def array_dirichlet_mutual_information(
    distribution: ArrayDirichletDistribution | np.ndarray, base: LogBase = None
) -> np.ndarray:
    """Compute the mutual information of a Dirichlet distribution."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    return array_dirichlet_entropy_of_expected_predictive_distribution(
        alphas, base=base
    ) - array_dirichlet_conditional_entropy(alphas, base=base)


@mutual_information.register(ArrayGaussianDistributionSample)
def array_gaussian_sample_mutual_information(
    sample: ArrayGaussianDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the epistemic uncertainty (total entropy minus aleatoric entropy)."""
    return array_gaussian_sample_entropy_of_expected_predictive_distribution(
        sample, base=base
    ) - array_gaussian_sample_conditional_entropy(sample, base=base)


@mutual_information.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_mutual_information(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the mutual information of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = np.mean(p, axis=axis, keepdims=True)

    if base == "normalize":
        base = p.shape[-1]

    return scipy_entropy(p, expected_value, axis=-1, base=base).mean(axis=axis)


# Zero-one proper scoring rule measures


@max_probability_complement_of_expected.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_max_probability_complement_of_expected(
    sample: ArrayCategoricalDistributionSample,
) -> np.ndarray:
    """Compute one minus the max probability of the expected value of a categorical sample."""
    expected_distribution = sample.sample_mean()
    return 1.0 - np.max(expected_distribution.probabilities, axis=-1)


@expected_max_probability_complement.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_expected_max_probability_complement(
    sample: ArrayCategoricalDistributionSample,
) -> np.ndarray:
    """Compute the expected value of one minus the max probability of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    per_sample_complement = 1.0 - np.max(p, axis=-1)
    return np.mean(per_sample_complement, axis=axis)


@expected_max_probability_complement.register(ArrayDirichletDistribution)
def array_dirichlet_expected_max_probability_complement(
    distribution: ArrayDirichletDistribution,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate ``1 - E[max_k p_k]`` for a Dirichlet by Monte-Carlo (no closed form)."""
    sample = distribution.sample(num_samples, rng=generator)
    return array_categorical_sample_expected_max_probability_complement(sample)


@max_disagreement.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_max_disagreement(
    sample: ArrayCategoricalDistributionSample,
) -> np.ndarray:
    """Compute the expected gap between each sample's max probability and its probability on the BMA argmax."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = np.mean(p, axis=axis, keepdims=True)
    bma_argmax = np.argmax(expected_value, axis=-1, keepdims=True)
    per_sample_bma_prob = np.take_along_axis(p, bma_argmax, axis=-1).squeeze(-1)
    per_sample_max = np.max(p, axis=-1)
    return np.mean(per_sample_max - per_sample_bma_prob, axis=axis)


# Generalized-entropy (proper scoring rule) measures


@generalized_entropy_of_expected.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_generalized_entropy_of_expected(
    sample: ArrayCategoricalDistributionSample, scoring_rule: ProperScoringRule
) -> np.ndarray:
    """Compute G(theta_bar) = <theta_bar, loss(theta_bar)> for a categorical sample."""
    mean = sample.sample_mean().probabilities  # (..., K)
    # 0 * inf = 0: a zero-probability outcome contributes nothing to the expected loss.
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = mean * scoring_rule.loss(mean)
    return np.where(mean > 0, weighted, 0.0).sum(axis=-1)


@expected_generalized_entropy.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_expected_generalized_entropy(
    sample: ArrayCategoricalDistributionSample, scoring_rule: ProperScoringRule
) -> np.ndarray:
    """Compute E[G(theta)] = mean_m <theta_m, loss(theta_m)> for a categorical sample."""
    p = sample.array.probabilities  # (..., M, K)
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    # 0 * inf = 0: a zero-probability outcome contributes nothing to the expected loss.
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = p * scoring_rule.loss(p)
    per_sample = np.where(p > 0, weighted, 0.0).sum(axis=-1)  # (..., M)
    return np.mean(per_sample, axis=axis)


# Distance-based epistemic uncertainty (Wasserstein)


def _min_expected_total_variation_from_samples(probabilities: np.ndarray, sample_axis: int) -> np.ndarray:
    """Solve ``1/2 min_q E_s ||p_s - q||_1`` over the simplex for a sample of distributions.

    Each ``q_k`` is the ``(1/2 - lambda)`` quantile of the marginal draws, where ``lambda`` is the
    single multiplier that makes ``q`` sum to one. The simplex sum is monotone in the quantile
    level, so ``lambda`` is found by bisection.
    """
    probabilities = np.moveaxis(probabilities, sample_axis, -2)  # (..., num_samples, num_classes)
    num_samples = probabilities.shape[-2]
    num_classes = probabilities.shape[-1]
    batch_shape = probabilities.shape[:-2]
    sorted_probabilities = np.sort(probabilities, axis=-2)

    def quantile_at(level: np.ndarray) -> np.ndarray:
        position = level * (num_samples - 1)  # (...)
        lower = np.floor(position).astype(np.intp)
        upper = np.minimum(lower + 1, num_samples - 1)
        fraction = (position - lower)[..., None]  # (..., 1)
        lower_index = np.broadcast_to(lower[..., None, None], (*batch_shape, 1, num_classes))
        upper_index = np.broadcast_to(upper[..., None, None], (*batch_shape, 1, num_classes))
        value_lower = np.take_along_axis(sorted_probabilities, lower_index, axis=-2)[..., 0, :]  # (..., num_classes)
        value_upper = np.take_along_axis(sorted_probabilities, upper_index, axis=-2)[..., 0, :]
        return value_lower + fraction * (value_upper - value_lower)

    # sum_k q_k(level) increases with the quantile level, so bisect for sum == 1.
    low = np.zeros(batch_shape)
    high = np.ones(batch_shape)
    for _ in range(TOTAL_VARIATION_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        below_target = quantile_at(mid).sum(axis=-1) < 1.0
        low = np.where(below_target, mid, low)
        high = np.where(below_target, high, mid)
    optimal_q = quantile_at(0.5 * (low + high))  # (..., num_classes)
    distances = np.abs(probabilities - optimal_q[..., None, :])  # (..., num_samples, num_classes)
    return 0.5 * np.mean(distances, axis=-2).sum(axis=-1)


@min_expected_total_variation.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_min_expected_total_variation(
    sample: ArrayCategoricalDistributionSample,
) -> np.ndarray:
    """Compute the distance-based epistemic uncertainty of a categorical sample."""
    probabilities = sample.array.probabilities
    sample_axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    return _min_expected_total_variation_from_samples(probabilities, sample_axis)


@min_expected_total_variation.register(ArrayDirichletDistribution)
def array_dirichlet_min_expected_total_variation(
    distribution: ArrayDirichletDistribution,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate the distance-based epistemic uncertainty of a Dirichlet by Monte-Carlo."""
    sample = distribution.sample(num_samples, rng=generator)
    return array_categorical_sample_min_expected_total_variation(sample)


# Vacuity


@vacuity.register(ArrayDirichletDistribution)
def array_dirichlet_vacuity(distribution: ArrayDirichletDistribution | np.ndarray) -> np.ndarray:
    """Compute the vacuity K / alpha_0 of a Dirichlet distribution."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    num_classes = alphas.shape[-1]
    alpha_0 = np.sum(alphas, axis=-1)
    return np.asarray(num_classes / alpha_0)


@max_probability_complement_of_expected.register(ArrayDirichletDistribution)
def array_dirichlet_max_probability_complement_of_expected(
    distribution: ArrayDirichletDistribution | np.ndarray,
) -> np.ndarray:
    """Compute one minus the max probability of the mean of a Dirichlet distribution.

    Closed form: ``1 - max_c (alpha_c / alpha_0)``.
    """
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = np.sum(alphas, axis=-1, keepdims=True)
    mean = alphas / alpha_0
    return 1.0 - np.max(mean, axis=-1)


# Dempster-Shafer uncertainty


@dempster_shafer_uncertainty.register(ArrayGaussianDistribution)
def array_gaussian_dempster_shafer_uncertainty(
    distribution: ArrayGaussianDistribution,
    mean_field_factor: float = DEFAULT_MEAN_FIELD_FACTOR,
) -> np.ndarray:
    """Compute the Dempster-Shafer uncertainty of a Gaussian over logits."""
    mean = distribution.mean
    var = distribution.var
    del distribution  # Avoid keeping a reference to the distribution for memory efficiency

    num_classes = mean.shape[-1]
    adjusted = mean / np.sqrt(1.0 + mean_field_factor * var)
    return num_classes / (num_classes + np.sum(np.exp(adjusted), axis=-1))
