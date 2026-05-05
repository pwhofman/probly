"""NumPy implementation of Metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog

from probly.representation.conformal_set.array import ArrayIntervalConformalSet, ArrayOneHotConformalSet
from probly.representation.credal_set.array import ArrayConvexCredalSet, ArrayProbabilityIntervalsCredalSet

from ._common import (
    auc,
    average_precision_score,
    coverage,
    efficiency,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

if TYPE_CHECKING:
    from probly.representation.distribution import ArrayCategoricalDistribution


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


# --- Predicted-set metrics ----------------------------------------------------


def _interval_dominance_mask(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Boolean mask of classes whose upper probability exceeds the global lower max.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Boolean array of shape ``(..., C)`` indicating selected classes.
    """
    threshold = np.max(lower, axis=-1, keepdims=True)
    return upper >= threshold


@coverage.register(ArrayOneHotConformalSet)
def _coverage_array_one_hot_conformal_set(y_pred: ArrayOneHotConformalSet, y_true: np.ndarray) -> np.floating:
    """Coverage for a one-hot conformal set: true class is in the selected set."""
    mask = np.asarray(y_pred.array)
    indices = np.asarray(y_true, dtype=np.int64)[..., None]
    return np.mean(np.take_along_axis(mask, indices, axis=-1).squeeze(-1))


@efficiency.register(ArrayOneHotConformalSet)
def _efficiency_array_one_hot_conformal_set(y_pred: ArrayOneHotConformalSet) -> np.floating:
    """Average cardinality of a one-hot conformal set."""
    return np.mean(np.asarray(y_pred.array).sum(axis=-1))


@coverage.register(ArrayIntervalConformalSet)
def _coverage_array_interval_conformal_set(y_pred: ArrayIntervalConformalSet, y_true: np.ndarray) -> np.floating:
    """Coverage for an interval conformal set: ``lower <= y_true <= upper``."""
    arr = np.asarray(y_pred.array)
    y = np.asarray(y_true)
    return np.mean((y >= arr[..., 0]) & (y <= arr[..., 1]))


@efficiency.register(ArrayIntervalConformalSet)
def _efficiency_array_interval_conformal_set(y_pred: ArrayIntervalConformalSet) -> np.floating:
    """Average width ``upper - lower`` of an interval conformal set."""
    arr = np.asarray(y_pred.array)
    return np.mean(arr[..., 1] - arr[..., 0])


@coverage.register(ArrayConvexCredalSet)
def _coverage_array_convex_credal_set(
    y_pred: ArrayConvexCredalSet, y_true: ArrayCategoricalDistribution
) -> np.floating:
    """Coverage for a convex credal set: target lies in the convex hull of the vertices.

    For each instance, runs the LP feasibility test
    ``V^T lambda = t, sum(lambda) = 1, lambda in [0, 1]``. Coverage is the
    fraction of instances where the LP is feasible.
    """
    vertices = np.asarray(y_pred.array.probabilities)
    targets = np.asarray(y_true.probabilities)
    n_instances, n_vertices, _ = vertices.shape
    c = np.zeros(n_vertices)
    bounds = [(0.0, 1.0)] * n_vertices
    covered = 0
    for i in range(n_instances):
        a_eq = np.vstack([vertices[i].T, np.ones(n_vertices)])
        b_eq = np.concatenate([targets[i], [1.0]])
        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds)
        covered += int(bool(res.success))
    return np.float64(covered / n_instances)


@efficiency.register(ArrayConvexCredalSet)
def _efficiency_array_convex_credal_set(y_pred: ArrayConvexCredalSet) -> np.floating:
    """Cardinality of the interval-dominance prediction set built from the vertex envelope."""
    mask = _interval_dominance_mask(y_pred.lower(), y_pred.upper())
    return np.mean(mask.sum(axis=-1))


@coverage.register(ArrayProbabilityIntervalsCredalSet)
def _coverage_array_probability_intervals_credal_set(
    y_pred: ArrayProbabilityIntervalsCredalSet,
    y_true: ArrayCategoricalDistribution,
) -> np.floating:
    """Coverage for a probability-intervals credal set: ``lower[k] <= target[k] <= upper[k]`` for all ``k``."""
    contained = y_pred.contains(np.asarray(y_true.probabilities))
    return np.mean(contained)


@efficiency.register(ArrayProbabilityIntervalsCredalSet)
def _efficiency_array_probability_intervals_credal_set(y_pred: ArrayProbabilityIntervalsCredalSet) -> np.floating:
    """Cardinality of the interval-dominance prediction set."""
    mask = _interval_dominance_mask(y_pred.lower(), y_pred.upper())
    return np.mean(mask.sum(axis=-1))
