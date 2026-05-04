"""NumPy implementation of Metrics."""

from __future__ import annotations

import numpy as np

from ._common import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@auc.register(np.ndarray)
def auc_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute area under a curve using the trapezoid rule."""
    return np.trapezoid(y, x, axis=-1)


@average_precision_score.register(np.ndarray)
def average_precision_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Compute average precision for NumPy arrays."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return -np.sum(np.diff(recall, axis=-1) * precision[..., :-1], axis=-1)  # ty:ignore[no-matching-overload, not-subscriptable]


@precision_recall_curve.register(np.ndarray)
def precision_recall_curve_numpy(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve along the last axis."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = y_score.shape[-1]

    desc_idx = np.flip(np.argsort(y_score, axis=-1, kind="mergesort"), axis=-1)
    y_score_sorted = np.take_along_axis(y_score, desc_idx, axis=-1)
    y_true_sorted = np.take_along_axis(y_true, desc_idx, axis=-1)

    tps = np.cumsum(y_true_sorted, axis=-1)
    predicted_pos = np.arange(1, n + 1, dtype=float)
    total_pos = tps[..., -1:]

    precision = tps / predicted_pos
    recall = np.where(total_pos > 0, tps / np.where(total_pos > 0, total_pos, 1.0), 0.0)

    ones = np.ones((*y_score.shape[:-1], 1))
    zeros = np.zeros((*y_score.shape[:-1], 1))
    precision = np.concatenate([np.flip(precision, axis=-1), ones], axis=-1)
    recall = np.concatenate([np.flip(recall, axis=-1), zeros], axis=-1)

    return precision, recall, y_score_sorted


@roc_auc_score.register(np.ndarray)
def roc_auc_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Compute area under the ROC curve for NumPy arrays."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)  # ty:ignore[invalid-return-type]


@roc_curve.register(np.ndarray)
def roc_curve_numpy(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve along the last axis."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = y_score.shape[-1]

    desc_idx = np.flip(np.argsort(y_score, axis=-1, kind="mergesort"), axis=-1)
    y_score_sorted = np.take_along_axis(y_score, desc_idx, axis=-1)
    y_true_sorted = np.take_along_axis(y_true, desc_idx, axis=-1)

    tps = np.cumsum(y_true_sorted, axis=-1)
    fps = np.arange(1, n + 1, dtype=float) - tps

    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = np.where(total_pos > 0, tps / np.where(total_pos > 0, total_pos, 1.0), 0.0)
    fpr = np.where(total_neg > 0, fps / np.where(total_neg > 0, total_neg, 1.0), 0.0)

    zeros = np.zeros((*y_score.shape[:-1], 1))
    tpr = np.concatenate([zeros, tpr], axis=-1)
    fpr = np.concatenate([zeros, fpr], axis=-1)
    thresholds = np.concatenate([y_score_sorted[..., :1] + 1, y_score_sorted], axis=-1)

    return fpr, tpr, thresholds
