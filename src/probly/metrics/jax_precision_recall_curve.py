"""JAX implementation of precision-recall curve."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.metrics import precision_recall_curve


@precision_recall_curve.register(jax.Array)
def precision_recall_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute precision-recall curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)
    n = y_score.shape[-1]

    desc_idx = jnp.flip(jnp.argsort(y_score, axis=-1, stable=True), axis=-1)
    y_score_sorted = jnp.take_along_axis(y_score, desc_idx, axis=-1)
    y_true_sorted = jnp.take_along_axis(y_true, desc_idx, axis=-1)

    tps = jnp.cumsum(y_true_sorted, axis=-1)
    predicted_pos = jnp.arange(1, n + 1, dtype=jnp.float32)
    total_pos = tps[..., -1:]

    precision = tps / predicted_pos
    recall = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)

    ones = jnp.ones((*y_score.shape[:-1], 1), dtype=jnp.float32)
    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    precision = jnp.concatenate([jnp.flip(precision, axis=-1), ones], axis=-1)
    recall = jnp.concatenate([jnp.flip(recall, axis=-1), zeros], axis=-1)

    return precision, recall, y_score_sorted
