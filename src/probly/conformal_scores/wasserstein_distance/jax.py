"""JAX implementation for Wasserstein distance scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import wasserstein_distance_score_func


@wasserstein_distance_score_func.register(jax.Array)
def compute_wasserstein_distance_score_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Computes the Wasserstein distance score using JAX arrays.

    Args:
        y_pred: Predicted probability mass functions.
        y_true: True probability mass functions or integer labels.
    """
    y_pred_j = jnp.asarray(y_pred)
    y_true_j = jnp.asarray(y_true)

    if y_true_j.ndim == 1 or (y_true_j.shape[0] == 1 and y_true_j.size == y_pred_j.shape[0]):
        y_one_hot = jnp.zeros_like(y_pred_j)
        y_one_hot = y_one_hot.at[jnp.arange(len(y_true_j)), y_true_j.flatten().astype(int)].set(1.0)
        y_true_j = y_one_hot

    return jnp.sum(jnp.abs(jnp.cumsum(y_pred_j, axis=-1) - jnp.cumsum(y_true_j, axis=-1)), axis=-1)
