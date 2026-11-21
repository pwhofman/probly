"""Jax implementations for ordinal classification measures."""

from __future__ import annotations

from jax import numpy as jnp

from probly.representation.distribution.jax_categorical import JaxCategoricalDistributionSample

from ._common import (
    LogBase,
    categorical_variance_aleatoric,
    categorical_variance_total,
)


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
