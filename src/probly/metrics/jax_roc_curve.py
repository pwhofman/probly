"""JAX implementation of ROC curve."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.metrics import roc_curve


@roc_curve.register(jax.Array)
def roc_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ROC curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)
    n = y_score.shape[-1]

    desc_idx = jnp.flip(jnp.argsort(y_score, axis=-1, stable=True), axis=-1)
    y_score_sorted = jnp.take_along_axis(y_score, desc_idx, axis=-1)
    y_true_sorted = jnp.take_along_axis(y_true, desc_idx, axis=-1)

    tps = jnp.cumsum(y_true_sorted, axis=-1)
    fps = jnp.arange(1, n + 1, dtype=jnp.float32) - tps

    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)
    fpr = jnp.where(total_neg > 0, fps / jnp.where(total_neg > 0, total_neg, 1.0), 0.0)

    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    tpr = jnp.concatenate([zeros, tpr], axis=-1)
    fpr = jnp.concatenate([zeros, fpr], axis=-1)
    thresholds = jnp.concatenate([y_score_sorted[..., :1] + 1, y_score_sorted], axis=-1)

    return fpr, tpr, thresholds
