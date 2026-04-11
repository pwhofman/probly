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

from ._common import conditional_entropy, entropy, entropy_of_expected_value, mutual_information

# Entropy


@entropy.register(ArrayCategoricalDistribution)
def array_categorical_entropy(distribution: ArrayCategoricalDistribution | np.ndarray) -> np.ndarray:
    """Compute the entropy of a categorical distribution represented as a numpy array."""
    if isinstance(distribution, ArrayCategoricalDistribution):
        p = distribution.probabilities
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        p = distribution
    return scipy_entropy(p, axis=-1)


@entropy.register(ArrayDirichletDistribution)
def array_dirichlet_entropy(distribution: ArrayDirichletDistribution | np.ndarray) -> np.ndarray:
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

    return log_beta + digamma_sum - digamma_individual


@entropy.register(ArrayGaussianDistribution)
def array_gaussian_entropy(distribution: ArrayGaussianDistribution | np.ndarray) -> np.ndarray:
    """Compute the (differential) entropy of a Gaussian distribution represented as a numpy array."""
    if isinstance(distribution, ArrayGaussianDistribution):
        var = distribution.var
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        var = distribution
    return 0.5 * np.log(2 * np.e * np.pi * var)


# Entropy of expected value


@entropy_of_expected_value.register(ArrayDirichletDistribution)
def array_dirichlet_entropy_of_expected_value(distribution: ArrayDirichletDistribution | np.ndarray) -> np.ndarray:
    """Compute the entropy of the expected value of a Dirichlet distribution."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    expected_value = alphas / np.sum(alphas, axis=-1, keepdims=True)
    return array_categorical_entropy(expected_value)


@entropy_of_expected_value.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_entropy_of_expected_value(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = np.mean(p, axis=axis)
    return array_categorical_entropy(expected_value)


# Conditional entropy


@conditional_entropy.register(ArrayDirichletDistribution)
def array_dirichlet_conditional_entropy(distribution: ArrayDirichletDistribution | np.ndarray) -> np.ndarray:
    """Compute the conditional entropy of a Dirichlet distribution."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = np.sum(alphas, axis=-1, keepdims=True)
    mean = alphas / alpha_0

    return special.digamma(alpha_0 + 1.0).squeeze(-1) - np.sum(mean * special.digamma(alphas + 1.0), axis=-1)


@conditional_entropy.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_conditional_entropy(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """Compute the conditional entropy of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    entropies = array_categorical_entropy(p)
    return np.mean(entropies, axis=axis)


# Mutual information


@mutual_information.register(ArrayDirichletDistribution)
def array_dirichlet_mutual_information(distribution: ArrayDirichletDistribution | np.ndarray) -> np.ndarray:
    """Compute the mutual information of a Dirichlet distribution."""
    if isinstance(distribution, ArrayDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    return array_dirichlet_entropy_of_expected_value(alphas) - array_dirichlet_conditional_entropy(alphas)


@mutual_information.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_mutual_information(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """Compute the mutual information of a sample from a categorical distribution."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = np.mean(p, axis=axis, keepdims=True)
    return scipy_entropy(p, expected_value, axis=-1).mean(axis=axis)
