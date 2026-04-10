"""NumPy implementation of ROC curve."""

from __future__ import annotations

import numpy as np

from ._common import roc_curve


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
