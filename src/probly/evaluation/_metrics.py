"""Numpy-based implementations of common classification metrics.

Replaces the subset of ``sklearn.metrics`` previously used by the evaluation
modules, removing the hard dependency on scikit-learn.
"""

from __future__ import annotations

import numpy as np


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area under a curve using the trapezoid rule.

    Args:
        x: x-coordinates (must be monotonic).
        y: y-coordinates.

    Returns:
        Area under the curve.
    """
    return float(np.trapezoid(y, x))


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute receiver operating characteristic (ROC) curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores (higher means more likely positive).

    Returns:
        fpr: False positive rates.
        tpr: True positive rates.
        thresholds: Decreasing score thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

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

    # Prepend (0, 0) point
    tpr = np.concatenate([np.array([0.0]), tpr])
    fpr = np.concatenate([np.array([0.0]), fpr])
    thresholds = np.concatenate([np.array([y_score_sorted[0] + 1]), y_score_sorted[threshold_idx]])

    return fpr, tpr, thresholds


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores (higher means more likely positive).

    Returns:
        precision: Precision values (ends with 1).
        recall: Recall values (ends with 0).
        thresholds: Decreasing score thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

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

    # Append (recall=0, precision=1) sentinel — sklearn convention
    precision = np.concatenate([precision, np.array([1.0])])
    recall = np.concatenate([recall, np.array([0.0])])

    # Reverse so recall is decreasing (sklearn convention)
    precision = precision[::-1]
    recall = recall[::-1]

    return precision, recall, y_score_sorted[threshold_idx]


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the ROC curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores.

    Returns:
        AUROC value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision (area under the precision-recall curve).

    Uses the step-function interpolation (sklearn convention).

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores.

    Returns:
        Average precision value.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(-np.sum(np.diff(recall) * precision[:-1]))
