"""Numpy implementations for ordinal classification measures."""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy

from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)

from ._common import (
    LogBase,
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_integer_variance_aleatoric,
    ordinal_integer_variance_total,
    ordinal_mutual_information_entropy,
    ordinal_mutual_information_variance,
    ordinal_variance,
    ordinal_variance_of_expected_predictive_distribution,
)


def _array_binary_entropy(p: np.ndarray, base: LogBase = None) -> np.ndarray:
    """Compute the binary Shannon entropy of probabilities ``p``.

    ``base="normalize"`` normalizes by ``log(2)`` so that the entropy is in
    ``[0, 1]`` for each binary problem.
    """
    scipy_base: float | None = 2.0 if base == "normalize" else base
    stacked = np.stack([p, 1.0 - p], axis=-1)
    return scipy_entropy(stacked, axis=-1, base=scipy_base)


def _cdf(p: np.ndarray) -> np.ndarray:
    """Compute the cumulative distribution function (CDF) excluding the last bin."""
    return np.cumsum(p, axis=-1)[..., :-1]


@ordinal_variance.register(ArrayCategoricalDistribution)
def array_categorical_ordinal_variance(
    distribution: ArrayCategoricalDistribution | np.ndarray,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the ordinal variance of a categorical distribution."""
    if isinstance(distribution, ArrayCategoricalDistribution):
        p = distribution.probabilities
        del distribution
    else:
        p = distribution
    cdf = _cdf(p)
    return np.sum(cdf * (1 - cdf), axis=-1)


@ordinal_entropy.register(ArrayCategoricalDistribution)
def array_categorical_ordinal_entropy(
    distribution: ArrayCategoricalDistribution | np.ndarray, base: LogBase = None
) -> np.ndarray:
    """Compute the ordinal entropy of a categorical distribution."""
    if isinstance(distribution, ArrayCategoricalDistribution):
        p = distribution.probabilities
        del distribution
    else:
        p = distribution
    cdf = _cdf(p)
    binary_entropies = _array_binary_entropy(cdf, base=base)
    return np.sum(binary_entropies, axis=-1)


@ordinal_variance_of_expected_predictive_distribution.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_ordinal_variance_of_expected_predictive_distribution(
    sample: ArrayCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the ordinal variance of the expected value of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    expected_cdf = np.mean(cdf, axis=axis)
    return np.sum(expected_cdf * (1 - expected_cdf), axis=-1)


@ordinal_conditional_variance.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_ordinal_conditional_variance(
    sample: ArrayCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the ordinal conditional variance of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    per_sample_variance = np.sum(cdf * (1 - cdf), axis=-1)
    return np.mean(per_sample_variance, axis=axis)


@ordinal_mutual_information_variance.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_ordinal_mutual_information_variance(
    sample: ArrayCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the ordinal mutual information (variance-based) of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    return np.sum(np.var(cdf, axis=axis, ddof=0), axis=-1)


@ordinal_entropy_of_expected_predictive_distribution.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_ordinal_entropy_of_expected_predictive_distribution(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the ordinal entropy of the expected value of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    expected_cdf = np.mean(cdf, axis=axis)
    binary_entropies = _array_binary_entropy(expected_cdf, base=base)
    return np.sum(binary_entropies, axis=-1)


@ordinal_conditional_entropy.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_ordinal_conditional_entropy(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the ordinal conditional entropy of a categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    binary_entropies = _array_binary_entropy(cdf, base=base)
    per_sample_entropy = np.sum(binary_entropies, axis=-1)
    return np.mean(per_sample_entropy, axis=axis)


@ordinal_mutual_information_entropy.register(ArrayCategoricalDistributionSample)
def array_categorical_sample_ordinal_mutual_information_entropy(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Compute the ordinal mutual information (entropy-based) of a categorical sample."""
    return array_categorical_sample_ordinal_entropy_of_expected_predictive_distribution(
        sample, base
    ) - array_categorical_sample_ordinal_conditional_entropy(sample, base)


def _integer_labels(num_classes: int) -> np.ndarray:
    """Integer encoding ``1, ..., K`` as a 1-D float array."""
    return np.arange(1, num_classes + 1, dtype=float)


@ordinal_integer_variance_total.register(ArrayCategoricalDistributionSample)
def array_ordinal_integer_variance_total(
    sample: ArrayCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the total variance under integer label encoding for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1])
    expected_p = np.mean(p, axis=axis)
    mu = np.sum(labels * expected_p, axis=-1, keepdims=True)
    return np.sum(((labels - mu) ** 2) * expected_p, axis=-1)


@ordinal_integer_variance_aleatoric.register(ArrayCategoricalDistributionSample)
def array_ordinal_integer_variance_aleatoric(
    sample: ArrayCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the aleatoric variance under integer label encoding for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1])
    mu_m = np.sum(labels * p, axis=-1, keepdims=True)
    per_model = np.sum(((labels - mu_m) ** 2) * p, axis=-1)
    return np.mean(per_model, axis=axis)
