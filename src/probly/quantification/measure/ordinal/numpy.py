"""Numpy implementations for ordinal classification measures."""

from __future__ import annotations

import numpy as np

from probly.representation.distribution.numpy_categorical import NumpyCategoricalDistributionSample

from ._common import (
    LogBase,
    categorical_variance_aleatoric,
    categorical_variance_total,
)


def _integer_labels(num_classes: int) -> np.ndarray:
    """Integer encoding ``1, ..., K`` as a 1-D float array."""
    return np.arange(1, num_classes + 1, dtype=float)


@categorical_variance_total.register(NumpyCategoricalDistributionSample)
def numpy_ordinal_integer_variance_total(
    sample: NumpyCategoricalDistributionSample,
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


@categorical_variance_aleatoric.register(NumpyCategoricalDistributionSample)
def numpy_ordinal_integer_variance_aleatoric(
    sample: NumpyCategoricalDistributionSample,
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
