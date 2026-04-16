"""Entropy measures for numpy array distributions."""

from __future__ import annotations

import numpy as np
from scipy import special
from scipy.stats import entropy as scipy_entropy

from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution
from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution

from ._common import (
    LogBase,
    conditional_entropy,
    entropy,
    entropy_of_expected_value,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    mutual_information,
)

# Entropy


@entropy.register(ArrayCategoricalDistribution)
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
    """Compute the (differential) entropy of a Gaussian distribution represented as a numpy array."""
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


@entropy_of_expected_value.register(ArrayDirichletDistribution)
def array_dirichlet_entropy_of_expected_value(
    distribution: ArrayDirichletDistribution | np.ndarray, base: LogBase = None
) -> np.ndarray:
    """Compute the entropy of the expected value of a Dirichlet distribution."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    expected_value = alphas / np.sum(alphas, axis=-1, keepdims=True)
    return array_categorical_entropy(expected_value, base=base)


@entropy_of_expected_value.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_entropy_of_expected_value(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = np.mean(p, axis=axis)
    return array_categorical_entropy(expected_value, base=base)


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

    return array_dirichlet_entropy_of_expected_value(alphas, base=base) - array_dirichlet_conditional_entropy(
        alphas, base=base
    )


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
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = np.mean(p, axis=axis)
    return 1.0 - np.max(expected_value, axis=-1)


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
