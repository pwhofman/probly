"""Numpy implementations of ordinal uncertainty decompositions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import entropy as scipy_entropy

from probly.representation.distribution.array_categorical import ArrayCategoricalDistributionSample
from probly.representation.distribution.array_gaussian import ArrayGaussianDistributionSample

from ._common import (
    categorical_variance_aleatoric,
    categorical_variance_total,
    gaussian_variance_aleatoric,
    gaussian_variance_epistemic,
    labelwise_binary_entropy_aleatoric,
    labelwise_binary_entropy_total,
    labelwise_binary_variance_aleatoric,
    labelwise_binary_variance_total,
    ordinal_binary_entropy_aleatoric,
    ordinal_binary_entropy_total,
    ordinal_binary_variance_aleatoric,
    ordinal_binary_variance_total,
)

if TYPE_CHECKING:
    from probly.quantification.measure.distribution._common import LogBase


def _array_binary_entropy(p: np.ndarray, base: LogBase = None) -> np.ndarray:
    """Compute the binary Shannon entropy of probabilities ``p``.

    ``base="normalize"`` normalizes by ``log(2)`` so that the entropy is in
    ``[0, 1]`` for each binary problem.
    """
    scipy_base: float | None
    if base == "normalize":
        scipy_base = 2.0
    else:
        scipy_base = base
    stacked = np.stack([p, 1.0 - p], axis=-1)
    return scipy_entropy(stacked, axis=-1, base=scipy_base)


def _cumulative_lower(p: np.ndarray) -> np.ndarray:
    """Return ``p_<=k`` for ``k = 1, ..., K-1`` along the trailing class axis."""
    cum = np.cumsum(p, axis=-1)
    return cum[..., :-1]


# OCS (Order-Consistent Split) binary reduction


@ordinal_binary_entropy_total.register(ArrayCategoricalDistributionSample)
def array_ordinal_binary_entropy_total(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """OCS binary-entropy total uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    expected_cum = np.mean(cum, axis=axis)
    binary_h = _array_binary_entropy(expected_cum, base=base)
    return np.sum(binary_h, axis=-1)


@ordinal_binary_entropy_aleatoric.register(ArrayCategoricalDistributionSample)
def array_ordinal_binary_entropy_aleatoric(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """OCS binary-entropy aleatoric uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    binary_h = _array_binary_entropy(cum, base=base)
    per_model_sum = np.sum(binary_h, axis=-1)
    return np.mean(per_model_sum, axis=axis)


@ordinal_binary_variance_total.register(ArrayCategoricalDistributionSample)
def array_ordinal_binary_variance_total(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """OCS binary-variance total uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    expected_cum = np.mean(cum, axis=axis)
    return np.sum(expected_cum * (1.0 - expected_cum), axis=-1)


@ordinal_binary_variance_aleatoric.register(ArrayCategoricalDistributionSample)
def array_ordinal_binary_variance_aleatoric(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """OCS binary-variance aleatoric uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    per_model = np.sum(cum * (1.0 - cum), axis=-1)
    return np.mean(per_model, axis=axis)


# Label-wise (one-vs-rest) binary reduction


@labelwise_binary_entropy_total.register(ArrayCategoricalDistributionSample)
def array_labelwise_binary_entropy_total(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Label-wise binary-entropy total uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    expected_p = np.mean(p, axis=axis)
    binary_h = _array_binary_entropy(expected_p, base=base)
    return np.sum(binary_h, axis=-1)


@labelwise_binary_entropy_aleatoric.register(ArrayCategoricalDistributionSample)
def array_labelwise_binary_entropy_aleatoric(
    sample: ArrayCategoricalDistributionSample, base: LogBase = None
) -> np.ndarray:
    """Label-wise binary-entropy aleatoric uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    binary_h = _array_binary_entropy(p, base=base)
    per_model_sum = np.sum(binary_h, axis=-1)
    return np.mean(per_model_sum, axis=axis)


@labelwise_binary_variance_total.register(ArrayCategoricalDistributionSample)
def array_labelwise_binary_variance_total(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """Label-wise binary-variance total uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    expected_p = np.mean(p, axis=axis)
    return np.sum(expected_p * (1.0 - expected_p), axis=-1)


@labelwise_binary_variance_aleatoric.register(ArrayCategoricalDistributionSample)
def array_labelwise_binary_variance_aleatoric(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """Label-wise binary-variance aleatoric uncertainty for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    per_model = np.sum(p * (1.0 - p), axis=-1)
    return np.mean(per_model, axis=axis)


# Standard categorical variance via law of total variance


def _integer_labels(num_classes: int) -> np.ndarray:
    """Integer encoding ``1, ..., K`` as a 1-D float array."""
    return np.arange(1, num_classes + 1, dtype=float)


@categorical_variance_total.register(ArrayCategoricalDistributionSample)
def array_categorical_variance_total(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """Total variance under integer label encoding for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1])
    expected_p = np.mean(p, axis=axis)
    mu = np.sum(labels * expected_p, axis=-1, keepdims=True)
    return np.sum(((labels - mu) ** 2) * expected_p, axis=-1)


@categorical_variance_aleatoric.register(ArrayCategoricalDistributionSample)
def array_categorical_variance_aleatoric(sample: ArrayCategoricalDistributionSample) -> np.ndarray:
    """Aleatoric variance under integer label encoding for a numpy categorical sample."""
    p = sample.array.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1])
    mu_m = np.sum(labels * p, axis=-1, keepdims=True)
    per_model = np.sum(((labels - mu_m) ** 2) * p, axis=-1)
    return np.mean(per_model, axis=axis)


# Gaussian regression variance via law of total variance


@gaussian_variance_aleatoric.register(ArrayGaussianDistributionSample)
def array_gaussian_variance_aleatoric(sample: ArrayGaussianDistributionSample) -> np.ndarray:
    """Aleatoric variance for a numpy Gaussian sample."""
    axis = sample.sample_axis
    return np.mean(sample.array.var, axis=axis)


@gaussian_variance_epistemic.register(ArrayGaussianDistributionSample)
def array_gaussian_variance_epistemic(sample: ArrayGaussianDistributionSample) -> np.ndarray:
    """Epistemic variance (variance of the predicted means) for a numpy Gaussian sample."""
    axis = sample.sample_axis
    return np.var(sample.array.mean, axis=axis, ddof=0)
