"""JAX implementation for Total Variation scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution
from probly.representation.sample.jax import JaxArraySample

from ._common import tv_score_func


@tv_score_func.register(jax.Array)
def compute_tv_score_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Computes the Total Variation score using JAX arrays."""
    y_pred_j = jnp.asarray(y_pred)
    y_true_j = jnp.asarray(y_true)

    if y_true_j.ndim == 1 or (y_true_j.shape[0] == 1 and y_true_j.size == y_pred_j.shape[0]):
        y_one_hot = jnp.zeros_like(y_pred_j)
        y_one_hot = y_one_hot.at[jnp.arange(len(y_true_j)), y_true_j.flatten().astype(int)].set(1.0)
        y_true_j = y_one_hot

    return 0.5 * jnp.sum(jnp.abs(y_pred_j - y_true_j), axis=-1)


@tv_score_func.register(JaxArraySample)
def _(y_pred: JaxArraySample, y_true: jax.Array) -> jax.Array:
    """Compute total variation scores for JAX samples."""
    return tv_score_func(y_pred.array, y_true)


@tv_score_func.register(JaxCategoricalDistribution)
def _(y_pred: JaxCategoricalDistribution, y_true: jax.Array) -> jax.Array:
    """Compute total variation scores for JAX categorical distributions."""
    return tv_score_func(y_pred.probabilities, y_true)
