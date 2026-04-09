"""NumPy implementation of ROC curve."""

from __future__ import annotations

import numpy as np

from ._common import roc_curve


@roc_curve.register(np.ndarray)
def roc_curve_numpy(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve for NumPy arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim == 2:
        return _roc_curve_numpy_batched(y_true, y_score)

    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    threshold_idx = np.concatenate([distinct_idx, np.array([len(y_true) - 1])])

    tps = np.cumsum(y_true_sorted)[threshold_idx]
    fps = threshold_idx + 1 - tps

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps, dtype=float)

    tpr = np.concatenate([np.array([0.0]), tpr])
    fpr = np.concatenate([np.array([0.0]), fpr])
    thresholds = np.concatenate([np.array([y_score_sorted[0] + 1]), y_score_sorted[threshold_idx]])

    return fpr, tpr, thresholds


def _roc_curve_numpy_batched(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_idx = np.arange(y_true.shape[0])[:, None]
    n = y_true.shape[1]

    desc_idx = np.argsort(y_score, axis=-1, kind="mergesort")[:, ::-1]
    y_score_sorted = y_score[batch_idx, desc_idx]
    y_true_sorted = y_true[batch_idx, desc_idx]

    tps = np.cumsum(y_true_sorted, axis=-1)
    fps = np.arange(1, n + 1)[None, :] - tps

    total_pos = y_true.sum(axis=-1, keepdims=True)
    total_neg = n - total_pos

    tpr = np.where(total_pos > 0, tps / total_pos, 0.0)
    fpr = np.where(total_neg > 0, fps / total_neg, 0.0)

    zeros = np.zeros((y_true.shape[0], 1))
    tpr = np.concatenate([zeros, tpr], axis=-1)
    fpr = np.concatenate([zeros, fpr], axis=-1)
    thresholds = np.concatenate([y_score_sorted[:, :1] + 1, y_score_sorted], axis=-1)

    return fpr, tpr, thresholds
