"""JAX implementation of Metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import (
    auc,
    average_precision_score,
    classwise_ece,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@classwise_ece.register(jax.Array)
def classwise_ece_jax(y_true: jax.Array, y_prob: jax.Array, *, num_bins: int = 15) -> jax.Array:
    """Compute the classwise expected calibration error for JAX arrays."""
    probs = y_prob.astype(jnp.float32)
    if probs.ndim != 2:
        msg = f"classwise_ece expects probabilities of shape (n, k), got shape {probs.shape}."
        raise ValueError(msg)
    labels = y_true.reshape(-1)
    n, k = probs.shape
    if labels.shape[0] != n:
        msg = f"classwise_ece labels must match probabilities batch size. Got {labels.shape[0]} labels for {n} rows."
        raise ValueError(msg)

    one_hot = (labels[:, None] == jnp.arange(k)[None, :]).astype(jnp.float32)
    bin_idx = jnp.minimum((probs * num_bins).astype(jnp.int32), num_bins - 1)

    total = jnp.float32(0.0)
    for b in range(num_bins):
        mask = (bin_idx == b).astype(jnp.float32)
        count = mask.sum(axis=0)
        safe_count = jnp.where(count > 0, count, 1.0)
        mean_prob = (probs * mask).sum(axis=0) / safe_count
        freq = (one_hot * mask).sum(axis=0) / safe_count
        total = total + (count / n * jnp.abs(freq - mean_prob)).sum()
    return total / k


@auc.register(jax.Array)
def auc_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute area under a curve using the trapezoid rule."""
    return jnp.trapezoid(y, x, axis=-1)


@average_precision_score.register(jax.Array)
def average_precision_score_jax(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
    """Compute average precision for JAX arrays."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return -jnp.sum(jnp.diff(recall, axis=-1) * precision[..., :-1], axis=-1)  # ty:ignore[invalid-argument-type, not-subscriptable]


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


@roc_auc_score.register(jax.Array)
def roc_auc_score_jax(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
    """Compute area under the ROC curve for JAX arrays."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)  # ty:ignore[invalid-return-type]
