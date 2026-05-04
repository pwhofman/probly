"""NumPy implementation of Metrics."""

from __future__ import annotations

import numpy as np

from probly.representation.conformal_set.array import ArrayIntervalConformalSet, ArrayOneHotConformalSet
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

from ._common import (
    auc,
    average_interval_width,
    average_precision_score,
    coverage,
    efficiency,
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


# --- Predicted-set metrics ----------------------------------------------------


def _interval_dominance_mask(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return the boolean mask of classes selected by the interval-dominance rule.

    A class ``y`` is selected when its upper probability is at least the
    maximum lower probability across all classes; equivalently, when no other
    class strictly dominates it under the credal set's lower/upper envelope.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Boolean array of shape ``(..., C)`` indicating selected classes.
    """
    threshold = np.max(lower, axis=-1, keepdims=True)
    return upper >= threshold


def _onehot_membership(mask: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Look up the membership flag for the true class along the last axis.

    Args:
        mask: Boolean class-membership mask of shape ``(..., C)``.
        y_true: Integer class labels of shape ``(...,)``.

    Returns:
        Boolean array of shape ``(...,)`` with ``True`` where the true class
        is in the predicted set.
    """
    indices = np.asarray(y_true, dtype=np.int64)[..., None]
    return np.take_along_axis(mask, indices, axis=-1).squeeze(-1)


def _envelope_coverage(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> np.floating:
    mask = _interval_dominance_mask(lower, upper)
    return np.mean(_onehot_membership(mask, np.asarray(y_true)))


def _envelope_efficiency(lower: np.ndarray, upper: np.ndarray) -> np.floating:
    mask = _interval_dominance_mask(lower, upper)
    return np.mean(mask.sum(axis=-1))


def _envelope_average_interval_width(lower: np.ndarray, upper: np.ndarray) -> np.floating:
    return np.mean(np.asarray(upper) - np.asarray(lower))


@coverage.register(ArrayOneHotConformalSet)
def _coverage_array_onehot(y_pred: ArrayOneHotConformalSet, y_true: np.ndarray) -> np.floating:
    """Coverage for a one-hot conformal set."""
    return np.mean(_onehot_membership(np.asarray(y_pred.array), np.asarray(y_true)))


@efficiency.register(ArrayOneHotConformalSet)
def _efficiency_array_onehot(y_pred: ArrayOneHotConformalSet) -> np.floating:
    """Average cardinality of a one-hot conformal set."""
    return np.mean(np.asarray(y_pred.array).sum(axis=-1))


@coverage.register(ArrayIntervalConformalSet)
def _coverage_array_interval(y_pred: ArrayIntervalConformalSet, y_true: np.ndarray) -> np.floating:
    """Coverage for an interval conformal set."""
    arr = np.asarray(y_pred.array)
    y = np.asarray(y_true)
    return np.mean((y >= arr[..., 0]) & (y <= arr[..., 1]))


@efficiency.register(ArrayIntervalConformalSet)
def _efficiency_array_interval(y_pred: ArrayIntervalConformalSet) -> np.floating:
    """Average width of an interval conformal set."""
    arr = np.asarray(y_pred.array)
    return np.mean(arr[..., 1] - arr[..., 0])


@coverage.register(ArraySingletonCredalSet)
def _coverage_array_singleton(y_pred: ArraySingletonCredalSet, y_true: np.ndarray) -> np.floating:
    """Top-1 coverage for a singleton credal set (degenerate to argmax accuracy)."""
    probs = np.asarray(y_pred.array.probabilities)
    predicted = np.argmax(probs, axis=-1)
    return np.mean(predicted == np.asarray(y_true))


@efficiency.register(ArraySingletonCredalSet)
def _efficiency_array_singleton(_: ArraySingletonCredalSet) -> np.floating:
    """A singleton credal set always yields a single predicted class."""
    return np.float64(1.0)


@coverage.register(ArrayDiscreteCredalSet)
def _coverage_array_discrete(y_pred: ArrayDiscreteCredalSet, y_true: np.ndarray) -> np.floating:
    """Coverage for a discrete credal set: any vertex's argmax matches the true class."""
    probs = np.asarray(y_pred.array.probabilities)
    argmax_per_vertex = np.argmax(probs, axis=-1)
    y = np.asarray(y_true)[..., None]
    return np.mean(np.any(argmax_per_vertex == y, axis=-1))


@efficiency.register(ArrayDiscreteCredalSet)
def _efficiency_array_discrete(y_pred: ArrayDiscreteCredalSet) -> np.floating:
    """Average number of distinct argmax classes across the vertex set."""
    probs = np.asarray(y_pred.array.probabilities)
    num_classes = probs.shape[-1]
    argmax_per_vertex = np.argmax(probs, axis=-1)
    classes_picked = (argmax_per_vertex[..., None] == np.arange(num_classes)).any(axis=-2)
    return np.mean(classes_picked.sum(axis=-1))


@coverage.register(ArrayConvexCredalSet)
def _coverage_array_convex(y_pred: ArrayConvexCredalSet, y_true: np.ndarray) -> np.floating:
    """Interval-dominance coverage for a convex credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayConvexCredalSet)
def _efficiency_array_convex(y_pred: ArrayConvexCredalSet) -> np.floating:
    """Interval-dominance prediction-set cardinality for a convex credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@coverage.register(ArrayDistanceBasedCredalSet)
def _coverage_array_distance(y_pred: ArrayDistanceBasedCredalSet, y_true: np.ndarray) -> np.floating:
    """Interval-dominance coverage for a distance-based credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayDistanceBasedCredalSet)
def _efficiency_array_distance(y_pred: ArrayDistanceBasedCredalSet) -> np.floating:
    """Interval-dominance prediction-set cardinality for a distance-based credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@coverage.register(ArrayProbabilityIntervalsCredalSet)
def _coverage_array_probability_intervals(
    y_pred: ArrayProbabilityIntervalsCredalSet, y_true: np.ndarray
) -> np.floating:
    """Interval-dominance coverage for a probability-intervals credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayProbabilityIntervalsCredalSet)
def _efficiency_array_probability_intervals(y_pred: ArrayProbabilityIntervalsCredalSet) -> np.floating:
    """Interval-dominance prediction-set cardinality for a probability-intervals credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@average_interval_width.register(ArrayConvexCredalSet)
def _average_interval_width_array_convex(y_pred: ArrayConvexCredalSet) -> np.floating:
    """Mean per-class width of the vertex-derived envelope of a convex credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(ArrayDiscreteCredalSet)
def _average_interval_width_array_discrete(y_pred: ArrayDiscreteCredalSet) -> np.floating:
    """Mean per-class width of the vertex-min/vertex-max envelope of a discrete credal set."""
    probs = np.asarray(y_pred.array.probabilities)
    return _envelope_average_interval_width(np.min(probs, axis=-2), np.max(probs, axis=-2))


@average_interval_width.register(ArrayDistanceBasedCredalSet)
def _average_interval_width_array_distance(y_pred: ArrayDistanceBasedCredalSet) -> np.floating:
    """Mean per-class width of the L1-clip envelope of a distance-based credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(ArrayProbabilityIntervalsCredalSet)
def _average_interval_width_array_probability_intervals(y_pred: ArrayProbabilityIntervalsCredalSet) -> np.floating:
    """Mean per-class interval width of a probability-intervals credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())
