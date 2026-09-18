"""JAX implementation of Metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import (
    auc,
    average_precision_score,
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


def _binary_clf_curve(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Count false and true positives at every score threshold, along the last axis.

    Adapted from scikit-learn's ``sklearn.metrics._ranking._binary_clf_curve`` (BSD-3-Clause). Samples are
    ranked by decreasing score and the counts after accepting each sample are recorded. Equal scores form a
    single threshold: sklearn keeps only the last index of every run of equal scores, which makes the number
    of thresholds depend on the data, whereas this batched version maps every sample to the last index of its
    run and keeps the ``(..., n)`` shape. For scores ``[0.9, 0.5, 0.5, 0.5, 0.2]`` the threshold indices are
    ``[0, 3, 3, 3, 4]``. Reading the counts per sample instead would let the sort order among tied samples
    decide the curve, and identical scores could then yield an AUROC of 1.0 or 0.0 instead of 0.5.

    Args:
        y_true: Binary labels of shape ``(..., n)``.
        y_score: Scores of shape ``(..., n)``.

    Returns:
        fps: False positives after each threshold, shape ``(..., n)``.
        tps: True positives after each threshold, shape ``(..., n)``.
        thresholds: Scores in decreasing order, shape ``(..., n)``.
    """
    n = y_score.shape[-1]
    # Sort scores and corresponding truth values.
    desc_score_indices = jnp.flip(jnp.argsort(y_score, axis=-1, stable=True), axis=-1)
    y_score = jnp.take_along_axis(y_score, desc_score_indices, axis=-1)
    y_true = jnp.take_along_axis(y_true, desc_score_indices, axis=-1)

    # y_score typically has many tied values. A distinct value ends where the next score differs, and the end
    # of the curve is always a threshold. The running minimum from the right turns these ends into the
    # threshold index of every position.
    end = jnp.ones((*y_score.shape[:-1], 1), dtype=bool)
    is_distinct_value = jnp.concatenate([y_score[..., 1:] != y_score[..., :-1], end], axis=-1)
    threshold_idxs = jnp.where(is_distinct_value, jnp.arange(n), n - 1)
    threshold_idxs = jax.lax.cummin(threshold_idxs, axis=threshold_idxs.ndim - 1, reverse=True)

    # Accumulate the true positives with decreasing threshold.
    tps = jnp.take_along_axis(jnp.cumsum(y_true, axis=-1), threshold_idxs, axis=-1)
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score


@precision_recall_curve.register(jax.Array)
def precision_recall_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute precision-recall curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]

    precision = tps / (tps + fps)
    recall = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)

    ones = jnp.ones((*y_score.shape[:-1], 1), dtype=jnp.float32)
    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    precision = jnp.concatenate([jnp.flip(precision, axis=-1), ones], axis=-1)
    recall = jnp.concatenate([jnp.flip(recall, axis=-1), zeros], axis=-1)

    return precision, recall, thresholds


@roc_curve.register(jax.Array)
def roc_curve_jax(y_true: jax.Array, y_score: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ROC curve along the last axis."""
    y_true = y_true.astype(jnp.float32)
    y_score = y_score.astype(jnp.float32)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = jnp.where(total_pos > 0, tps / jnp.where(total_pos > 0, total_pos, 1.0), 0.0)
    fpr = jnp.where(total_neg > 0, fps / jnp.where(total_neg > 0, total_neg, 1.0), 0.0)

    zeros = jnp.zeros((*y_score.shape[:-1], 1), dtype=jnp.float32)
    tpr = jnp.concatenate([zeros, tpr], axis=-1)
    fpr = jnp.concatenate([zeros, fpr], axis=-1)
    thresholds = jnp.concatenate([thresholds[..., :1] + 1, thresholds], axis=-1)

    return fpr, tpr, thresholds


@roc_auc_score.register(jax.Array)
def roc_auc_score_jax(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
    """Compute area under the ROC curve for JAX arrays."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)  # ty:ignore[invalid-return-type]
