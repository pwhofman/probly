"""JAX implementation for Inner Product scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import inner_product_score_func


@inner_product_score_func.register(jax.Array)
def compute_inner_product_score_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Computes the Inner Product score using JAX Array."""
    y_pred_t = jnp.asarray(y_pred)
    y_true_t = jnp.asarray(y_true)

    if y_true_t.ndim == 1 or (y_true_t.shape[0] == 1 and y_true_t.size == y_pred_t.shape[0]):
        y_one_hot = jnp.zeros_like(y_pred_t)
        y_one_hot = y_one_hot.at[jnp.arange(len(y_true_t)), y_true_t.flatten().astype(int)].set(1.0)
        y_true_t = y_one_hot

    return 1.0 - jnp.sum(y_pred_t * y_true_t, axis=-1)
