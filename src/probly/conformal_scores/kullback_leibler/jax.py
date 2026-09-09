"""JAX implementation for Kullback-Leibler divergence scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import kl_divergence_score_func


@kl_divergence_score_func.register(jax.Array)
def compute_kl_divergence_score_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Computes the Kullback-Leibler divergence score using JAX Array."""
    y_pred_t = jnp.asarray(y_pred)
    y_true_t = jnp.asarray(y_true)

    eps = 1e-12
    if y_true_t.shape == y_pred_t.shape[:-1] or (
        y_pred_t.ndim == 2 and y_true_t.shape == (1, y_pred_t.shape[0]) and y_true_t.shape != y_pred_t.shape
    ):
        labels = y_true_t.reshape(y_pred_t.shape[:-1]).astype(int)
        probabilities = jnp.take_along_axis(y_pred_t, labels[..., None], axis=-1).squeeze(-1)
        return -jnp.log(jnp.clip(probabilities, min=eps))

    y_pred_safe = jnp.clip(y_pred_t, min=eps)
    y_true_safe = jnp.clip(y_true_t, min=eps)

    return jnp.sum(y_true_t * jnp.log(y_true_safe / y_pred_safe), axis=-1)
