"""NumPy implementation of Metrics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog

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
    convex_hull_coverage,
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


def _credal_containment_coverage(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> np.floating:
    """Fraction of instances where ``y_true`` lies in ``[lower, upper]`` for all classes.

    Args:
        lower: Lower probability envelope of shape ``(N, C)``.
        upper: Upper probability envelope of shape ``(N, C)``.
        y_true: Target probability vectors of shape ``(N, C)``.

    Returns:
        Mean containment indicator as a scalar float.
    """
    y = np.asarray(y_true)
    covered = np.all((lower <= y) & (y <= upper), axis=-1)
    return np.mean(covered)


def _credal_interval_efficiency(lower: np.ndarray, upper: np.ndarray) -> np.floating:
    """Efficiency of a credal set as ``1 - mean(upper - lower)``.

    Args:
        lower: Lower probability envelope of shape ``(N, C)``.
        upper: Upper probability envelope of shape ``(N, C)``.

    Returns:
        Scalar in ``(-inf, 1]``; higher means a tighter (more efficient) credal set.
    """
    return np.float64(1.0 - float(np.mean(np.asarray(upper) - np.asarray(lower))))


@coverage.register(ArrayConvexCredalSet)
def _coverage_array_convex(y_pred: ArrayConvexCredalSet, y_true: np.ndarray) -> np.floating:
    """Containment coverage for a convex credal set.

    Args:
        y_pred: Convex credal set.
        y_true: Target probability vectors of shape ``(N, C)``.

    Returns:
        Fraction of instances where the target lies in ``[lower, upper]`` for all classes.
    """
    return _credal_containment_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayConvexCredalSet)
def _efficiency_array_convex(y_pred: ArrayConvexCredalSet) -> np.floating:
    """Interval-width efficiency for a convex credal set: ``1 - mean(upper - lower)``.

    Returns:
        Scalar efficiency; higher means a tighter credal set.
    """
    return _credal_interval_efficiency(y_pred.lower(), y_pred.upper())


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
    """Containment coverage for a probability-intervals credal set.

    Args:
        y_pred: Probability-intervals credal set.
        y_true: Target probability vectors of shape ``(N, C)``.

    Returns:
        Fraction of instances where the target lies in ``[lower, upper]`` for all classes.
    """
    return _credal_containment_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayProbabilityIntervalsCredalSet)
def _efficiency_array_probability_intervals(y_pred: ArrayProbabilityIntervalsCredalSet) -> np.floating:
    """Interval-width efficiency for a probability-intervals credal set: ``1 - mean(upper - lower)``.

    Returns:
        Scalar efficiency; higher means a tighter credal set.
    """
    return _credal_interval_efficiency(y_pred.lower(), y_pred.upper())


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


# --- Convex-hull coverage ----------------------------------------------------


def _validate_epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or epsilon < 0.0:
        msg = f"epsilon must be a non-negative finite float, got {epsilon!r}."
        raise ValueError(msg)


def _convex_hull_lp_coverage(
    vertices: np.ndarray,
    targets: np.ndarray,
    epsilon: float,
    **linprog_kwargs: object,
) -> np.floating:
    """Convex-hull membership coverage via per-instance LP feasibility.

    Solves one linear program per instance. The strict variant
    (``epsilon == 0``) tests feasibility of
    ``V^T lambda = t, sum(lambda) = 1, lambda in [0, 1]``. The relaxed
    variant (``epsilon > 0``) introduces L1 slack variables ``s+`` and
    ``s-`` and minimizes their sum; an instance counts as covered iff the
    LP is feasible and the optimal slack sum is at most ``epsilon``.

    Args:
        vertices: Array of shape ``(N, V, K)`` holding ``V`` vertex
            distributions over ``K`` classes for each of ``N`` instances.
        targets: Array of shape ``(N, K)`` holding the target distribution
            for each instance.
        epsilon: L1 tolerance. ``0.0`` selects the strict LP.
        **linprog_kwargs: Forwarded to :func:`scipy.optimize.linprog`
            (e.g. ``method`` or solver tolerances).

    Returns:
        Fraction of instances whose target lies in (or within ``epsilon`` of)
        the hull, as ``np.float64``.
    """
    _validate_epsilon(epsilon)
    if vertices.ndim != 3:
        msg = f"vertices must be 3D (N, V, K); got shape {vertices.shape}."
        raise ValueError(msg)
    if targets.ndim != 2:
        msg = f"targets must be 2D (N, K); got shape {targets.shape}."
        raise ValueError(msg)
    if vertices.shape[0] != targets.shape[0]:
        msg = f"vertices and targets must agree on N; got {vertices.shape[0]} and {targets.shape[0]}."
        raise ValueError(msg)
    if vertices.shape[2] != targets.shape[1]:
        msg = f"vertices and targets must agree on K; got {vertices.shape[2]} and {targets.shape[1]}."
        raise ValueError(msg)

    n_instances, n_vertices, n_classes = vertices.shape
    relaxed = epsilon > 0.0

    if relaxed:
        c = np.concatenate([np.zeros(n_vertices), np.ones(2 * n_classes)])
        bounds: list[tuple[float, float | None]] = [(0.0, 1.0)] * n_vertices + [(0.0, None)] * (2 * n_classes)
    else:
        c = np.zeros(n_vertices)
        bounds = [(0.0, 1.0)] * n_vertices

    covered = 0
    # Per-instance LP loop (Python-level). For very large N (~10^6) consider
    # joblib.Parallel; not implemented here to keep the dependency surface small.
    for i in range(n_instances):
        v = vertices[i]
        t = targets[i]
        if relaxed:
            a_eq_top = np.hstack([v.T, np.eye(n_classes), -np.eye(n_classes)])
            a_eq_bot = np.concatenate([np.ones(n_vertices), np.zeros(2 * n_classes)])
            a_eq = np.vstack([a_eq_top, a_eq_bot])
            b_eq = np.concatenate([t, [1.0]])
        else:
            a_eq = np.vstack([v.T, np.ones(n_vertices)])
            b_eq = np.concatenate([t, [1.0]])

        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, **linprog_kwargs)
        if relaxed:
            covered += int(bool(res.success) and float(res.fun) <= epsilon)
        else:
            covered += int(bool(res.success))

    return np.float64(covered / n_instances) if n_instances > 0 else np.float64("nan")


@convex_hull_coverage.register(ArrayConvexCredalSet)
def _convex_hull_coverage_array_convex(
    y_pred: ArrayConvexCredalSet,
    y_true: ArrayCategoricalDistribution,
    *,
    epsilon: float = 0.0,
    **linprog_kwargs: object,
) -> np.floating:
    """LP-based hull coverage for a convex credal set."""
    return _convex_hull_lp_coverage(
        np.asarray(y_pred.array.probabilities),
        np.asarray(y_true.probabilities),
        epsilon,
        **linprog_kwargs,
    )


@convex_hull_coverage.register(ArrayDiscreteCredalSet)
def _convex_hull_coverage_array_discrete(
    y_pred: ArrayDiscreteCredalSet,
    y_true: ArrayCategoricalDistribution,
    *,
    epsilon: float = 0.0,
    **linprog_kwargs: object,
) -> np.floating:
    """LP-based hull coverage for a discrete credal set (same vertex structure as Convex)."""
    return _convex_hull_lp_coverage(
        np.asarray(y_pred.array.probabilities),
        np.asarray(y_true.probabilities),
        epsilon,
        **linprog_kwargs,
    )


@convex_hull_coverage.register(ArraySingletonCredalSet)
def _convex_hull_coverage_array_singleton(
    y_pred: ArraySingletonCredalSet,
    y_true: ArrayCategoricalDistribution,
    *,
    epsilon: float = 0.0,
    **_linprog_kwargs: object,
) -> np.floating:
    """Hull degenerates to a point; coverage is closed-form L1 distance test.

    The singleton handler does not call ``linprog`` and is therefore unaffected
    by solver tolerances. ``epsilon=0.0`` performs strict element-wise equality
    of ``predicted == target`` (subject to float arithmetic), which can produce
    slightly different verdicts than the LP path on numerically-tight inputs.
    """
    _validate_epsilon(epsilon)
    predicted = np.asarray(y_pred.array.probabilities)
    targets = np.asarray(y_true.probabilities)
    l1_dist = np.abs(predicted - targets).sum(axis=-1)
    return np.mean(l1_dist <= epsilon)
