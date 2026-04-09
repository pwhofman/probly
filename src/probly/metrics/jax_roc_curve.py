"""JAX implementation of ROC curve."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.metrics import roc_curve


@roc_curve.register(jax.Array)
def roc_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ROC curve for JAX arrays."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)
    if y_true.ndim == 2:
        return _roc_curve_jax_batched(y_true, y_score)

    desc_idx = jnp.argsort(y_score, stable=True)[::-1]
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    distinct_idx = jnp.where(jnp.diff(y_score_sorted))[0]
    threshold_idx = jnp.concatenate([distinct_idx, jnp.array([len(y_true) - 1])])

    tps = jnp.cumsum(y_true_sorted)[threshold_idx]
    fps = (threshold_idx + 1).astype(jnp.float32) - tps

    total_pos = y_true.sum()
    total_neg = float(len(y_true)) - total_pos

    tpr = jnp.where(total_pos > 0, tps / total_pos, jnp.zeros_like(tps))
    fpr = jnp.where(total_neg > 0, fps / total_neg, jnp.zeros_like(fps))

    tpr = jnp.concatenate([jnp.array([0.0]), tpr])
    fpr = jnp.concatenate([jnp.array([0.0]), fpr])
    thresholds = jnp.concatenate([y_score_sorted[:1] + 1, y_score_sorted[threshold_idx]])

    return fpr, tpr, thresholds


def _roc_curve_jax_batched(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_idx = jnp.arange(y_true.shape[0])[:, None]
    n = y_true.shape[1]

    desc_idx = jnp.argsort(y_score, axis=-1, stable=True)[:, ::-1]
    y_score_sorted = y_score[batch_idx, desc_idx]
    y_true_sorted = y_true[batch_idx, desc_idx]

    tps = jnp.cumsum(y_true_sorted, axis=-1)
    fps = jnp.arange(1, n + 1, dtype=jnp.float32)[None, :] - tps

    total_pos = y_true.sum(axis=-1, keepdims=True)
    total_neg = n - total_pos

    tpr = jnp.where(total_pos > 0, tps / total_pos, 0.0)
    fpr = jnp.where(total_neg > 0, fps / total_neg, 0.0)

    zeros = jnp.zeros((y_true.shape[0], 1))
    tpr = jnp.concatenate([zeros, tpr], axis=-1)
    fpr = jnp.concatenate([zeros, fpr], axis=-1)
    thresholds = jnp.concatenate([y_score_sorted[:, :1] + 1, y_score_sorted], axis=-1)

    return fpr, tpr, thresholds
