"""NumPy implementation of precision-recall curve."""

from __future__ import annotations

import numpy as np

from ._common import precision_recall_curve


@precision_recall_curve.register(np.ndarray)
def precision_recall_curve_numpy(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve for NumPy arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim == 2:
        return _precision_recall_curve_numpy_batched(y_true, y_score)

    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    threshold_idx = np.concatenate([distinct_idx, np.array([len(y_true) - 1])])

    tps = np.cumsum(y_true_sorted)[threshold_idx]
    predicted_pos = threshold_idx + 1
    total_pos = y_true.sum()

    precision = tps / predicted_pos
    recall = tps / total_pos if total_pos > 0 else np.zeros_like(tps, dtype=float)

    precision = np.concatenate([precision[::-1], np.array([1.0])])
    recall = np.concatenate([recall[::-1], np.array([0.0])])

    return precision, recall, y_score_sorted[threshold_idx]


def _precision_recall_curve_numpy_batched(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_idx = np.arange(y_true.shape[0])[:, None]
    n = y_true.shape[1]

    desc_idx = np.argsort(y_score, axis=-1, kind="mergesort")[:, ::-1]
    y_score_sorted = y_score[batch_idx, desc_idx]
    y_true_sorted = y_true[batch_idx, desc_idx]

    tps = np.cumsum(y_true_sorted, axis=-1)
    predicted_pos = np.arange(1, n + 1, dtype=float)[None, :]
    total_pos = y_true.sum(axis=-1, keepdims=True)

    precision = tps / predicted_pos
    recall = np.where(total_pos > 0, tps / total_pos, 0.0)

    precision = np.concatenate([precision[:, ::-1], np.ones((y_true.shape[0], 1))], axis=-1)
    recall = np.concatenate([recall[:, ::-1], np.zeros((y_true.shape[0], 1))], axis=-1)

    return precision, recall, y_score_sorted
