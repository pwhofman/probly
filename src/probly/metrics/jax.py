"""JAX implementation of Metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import (
    auc,
    average_interval_size,
    average_precision_score,
    average_set_size,
    empirical_coverage_classification,
    empirical_coverage_regression,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


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


@empirical_coverage_classification.register(jnp.ndarray)
def _empirical_coverage_classification_jax(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    contained = y_pred[jnp.arange(len(y_true)), y_true.astype(int)]
    return contained.mean().item()


@empirical_coverage_regression.register(jnp.ndarray)
def _empirical_coverage_regression_jax(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> float:
    return ((y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1])).mean().item()


@average_set_size.register(jnp.ndarray)
def _average_set_size_jax(y_pred: jnp.ndarray) -> float:
    return y_pred.sum(axis=1).mean().item()


@average_interval_size.register(jnp.ndarray)
def _average_interval_size_jax(y_pred: jnp.ndarray) -> float:
    return (y_pred[:, 1] - y_pred[:, 0]).mean().item()
