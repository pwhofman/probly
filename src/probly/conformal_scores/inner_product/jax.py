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

    if y_true_t.shape == y_pred_t.shape[:-1] or (
        y_pred_t.ndim == 2 and y_true_t.shape == (1, y_pred_t.shape[0]) and y_true_t.shape != y_pred_t.shape
    ):
        labels = y_true_t.reshape(y_pred_t.shape[:-1]).astype(int)
        return 1.0 - jnp.take_along_axis(y_pred_t, labels[..., None], axis=-1).squeeze(-1)

    return 1.0 - jnp.sum(y_pred_t * y_true_t, axis=-1)
