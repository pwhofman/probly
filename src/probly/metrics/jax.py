"""JAX implementation of Metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import (
    accuracy,
    auc,
    average_precision_score,
    classwise_ece,
    expected_calibration_error,
    false_negative_rate,
    false_positive_rate,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@accuracy.register(jax.Array)
def accuracy_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Compute top-1 classification accuracy for JAX arrays."""
    labels = y_true.reshape(-1)
    predicted = y_pred
    if predicted.ndim == 2:
        predicted = jnp.argmax(predicted, axis=-1)
    elif predicted.ndim != 1:
        msg = f"accuracy expects predictions of shape (n,) or (n, k), got shape {predicted.shape}."
        raise ValueError(msg)
    if predicted.shape[0] != labels.shape[0]:
        msg = (
            "accuracy labels must match predictions batch size. "
            f"Got {labels.shape[0]} labels for {predicted.shape[0]} predictions."
        )
        raise ValueError(msg)
    return (predicted == labels).astype(jnp.float32).mean()


@classwise_ece.register(jax.Array)
def classwise_ece_jax(y_prob: jax.Array, y_true: jax.Array, *, num_bins: int = 15) -> jax.Array:
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
    if num_bins < 1:
        msg = f"classwise_ece expects num_bins >= 1, got {num_bins}."
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


@expected_calibration_error.register(jax.Array)
def expected_calibration_error_jax(y_prob: jax.Array, y_true: jax.Array, *, num_bins: int = 15) -> jax.Array:
    """Compute the confidence expected calibration error for JAX arrays."""
    probs = y_prob.astype(jnp.float32)
    if probs.ndim != 2:
        msg = f"expected_calibration_error expects probabilities of shape (n, k), got shape {probs.shape}."
        raise ValueError(msg)
    labels = y_true.reshape(-1)
    n = probs.shape[0]
    if labels.shape[0] != n:
        msg = (
            "expected_calibration_error labels must match probabilities batch size. "
            f"Got {labels.shape[0]} labels for {n} rows."
        )
        raise ValueError(msg)
    if num_bins < 1:
        msg = f"expected_calibration_error expects num_bins >= 1, got {num_bins}."
        raise ValueError(msg)

    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == labels).astype(jnp.float32)
    bin_idx = jnp.minimum((conf * num_bins).astype(jnp.int32), num_bins - 1)

    total = jnp.float32(0.0)
    for b in range(num_bins):
        mask = (bin_idx == b).astype(jnp.float32)
        count = mask.sum()
        safe_count = jnp.where(count > 0, count, 1.0)
        acc_bin = (correct * mask).sum() / safe_count
        conf_bin = (conf * mask).sum() / safe_count
        total = total + count / n * jnp.abs(acc_bin - conf_bin)
    return total


@false_positive_rate.register(jax.Array)
def false_positive_rate_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Compute the false positive rate for JAX arrays."""
    y = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    if p.shape[0] != y.shape[0]:
        msg = f"false_positive_rate y_true and y_pred must match in batch size. Got {y.shape[0]} and {p.shape[0]}."
        raise ValueError(msg)
    negatives = (y == 0).astype(jnp.float32)
    false_positives = (p == 1).astype(jnp.float32) * negatives
    # 0/0 yields NaN when there are no negative samples.
    return false_positives.sum() / negatives.sum()


@false_negative_rate.register(jax.Array)
def false_negative_rate_jax(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Compute the false negative rate for JAX arrays."""
    y = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    if p.shape[0] != y.shape[0]:
        msg = f"false_negative_rate y_true and y_pred must match in batch size. Got {y.shape[0]} and {p.shape[0]}."
        raise ValueError(msg)
    positives = (y == 1).astype(jnp.float32)
    false_negatives = (p == 0).astype(jnp.float32) * positives
    # 0/0 yields NaN when there are no positive samples.
    return false_negatives.sum() / positives.sum()


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
