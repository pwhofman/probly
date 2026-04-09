"""JAX implementation of precision-recall curve."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.metrics import precision_recall_curve


@precision_recall_curve.register(jax.Array)
def precision_recall_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute precision-recall curve for JAX arrays."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)
    if y_true.ndim == 2:
        return _precision_recall_curve_jax_batched(y_true, y_score)

    desc_idx = jnp.argsort(y_score, stable=True)[::-1]
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    distinct_idx = jnp.where(jnp.diff(y_score_sorted))[0]
    threshold_idx = jnp.concatenate([distinct_idx, jnp.array([len(y_true) - 1])])

    tps = jnp.cumsum(y_true_sorted)[threshold_idx]
    predicted_pos = (threshold_idx + 1).astype(jnp.float32)
    total_pos = y_true.sum()

    precision = tps / predicted_pos
    recall = jnp.where(total_pos > 0, tps / total_pos, jnp.zeros_like(tps))

    precision = jnp.concatenate([precision[::-1], jnp.array([1.0])])
    recall = jnp.concatenate([recall[::-1], jnp.array([0.0])])

    return precision, recall, y_score_sorted[threshold_idx]


def _precision_recall_curve_jax_batched(
    y_true: jax.Array, y_score: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_idx = jnp.arange(y_true.shape[0])[:, None]
    n = y_true.shape[1]

    desc_idx = jnp.argsort(y_score, axis=-1, stable=True)[:, ::-1]
    y_score_sorted = y_score[batch_idx, desc_idx]
    y_true_sorted = y_true[batch_idx, desc_idx]

    tps = jnp.cumsum(y_true_sorted, axis=-1)
    predicted_pos = jnp.arange(1, n + 1, dtype=jnp.float32)[None, :]
    total_pos = y_true.sum(axis=-1, keepdims=True)

    precision = tps / predicted_pos
    recall = jnp.where(total_pos > 0, tps / total_pos, 0.0)

    precision = jnp.concatenate([precision[:, ::-1], jnp.ones((y_true.shape[0], 1))], axis=-1)
    recall = jnp.concatenate([recall[:, ::-1], jnp.zeros((y_true.shape[0], 1))], axis=-1)

    return precision, recall, y_score_sorted
