"""NumPy implementations of :func:`coverage`, :func:`efficiency`, and :func:`average_interval_width`.

Registers the dispatch handlers for every NumPy-backed conformal-set and
credal-set representation. Coverage uses the membership-of-the-true-class
semantics for conformal sets, and a per-credal-set rule for credal sets:

* :class:`~probly.representation.credal_set.array.ArraySingletonCredalSet`
  collapses to top-1 accuracy.
* :class:`~probly.representation.credal_set.array.ArrayDiscreteCredalSet`
  is covered iff some vertex distribution puts its argmax on the true class.
* :class:`~probly.representation.credal_set.array.ArrayConvexCredalSet`,
  :class:`~probly.representation.credal_set.array.ArrayDistanceBasedCredalSet`
  and
  :class:`~probly.representation.credal_set.array.ArrayProbabilityIntervalsCredalSet`
  use the interval-dominance prediction set (a class is selected iff its
  upper probability is at least the maximum lower probability across all
  classes).

Note:
    The ``DistanceBased`` envelope (``clip(nominal +- radius, 0, 1)``) ignores
    the simplex constraint and is therefore loose. The interval-dominance
    rule applied to it is consistent with the declared lower/upper but is
    over-conservative relative to a tight L1-ball envelope.
"""

from __future__ import annotations

import numpy as np

from probly.evaluation.metrics import average_interval_width, coverage, efficiency
from probly.representation.conformal_set.array import ArrayIntervalConformalSet, ArrayOneHotConformalSet
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)


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


def _envelope_coverage(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(np.mean(_onehot_membership(mask, np.asarray(y_true))))


def _envelope_efficiency(lower: np.ndarray, upper: np.ndarray) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(np.mean(mask.sum(axis=-1)))


def _envelope_average_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


@coverage.register(ArrayOneHotConformalSet)
def _coverage_array_onehot(y_pred: ArrayOneHotConformalSet, y_true: np.ndarray) -> float:
    """Coverage for a one-hot conformal set."""
    return float(np.mean(_onehot_membership(np.asarray(y_pred.array), np.asarray(y_true))))


@efficiency.register(ArrayOneHotConformalSet)
def _efficiency_array_onehot(y_pred: ArrayOneHotConformalSet) -> float:
    """Average cardinality of a one-hot conformal set."""
    return float(np.mean(np.asarray(y_pred.array).sum(axis=-1)))


@coverage.register(ArrayIntervalConformalSet)
def _coverage_array_interval(y_pred: ArrayIntervalConformalSet, y_true: np.ndarray) -> float:
    """Coverage for an interval conformal set."""
    arr = np.asarray(y_pred.array)
    y = np.asarray(y_true)
    return float(np.mean((y >= arr[..., 0]) & (y <= arr[..., 1])))


@efficiency.register(ArrayIntervalConformalSet)
def _efficiency_array_interval(y_pred: ArrayIntervalConformalSet) -> float:
    """Average width of an interval conformal set."""
    arr = np.asarray(y_pred.array)
    return float(np.mean(arr[..., 1] - arr[..., 0]))


@coverage.register(ArraySingletonCredalSet)
def _coverage_array_singleton(y_pred: ArraySingletonCredalSet, y_true: np.ndarray) -> float:
    """Top-1 coverage for a singleton credal set (degenerate to argmax accuracy)."""
    probs = np.asarray(y_pred.array.probabilities)
    predicted = np.argmax(probs, axis=-1)
    return float(np.mean(predicted == np.asarray(y_true)))


@efficiency.register(ArraySingletonCredalSet)
def _efficiency_array_singleton(_: ArraySingletonCredalSet) -> float:
    """A singleton credal set always yields a single predicted class."""
    return 1.0


@coverage.register(ArrayDiscreteCredalSet)
def _coverage_array_discrete(y_pred: ArrayDiscreteCredalSet, y_true: np.ndarray) -> float:
    """Coverage for a discrete credal set: any vertex's argmax matches the true class."""
    probs = np.asarray(y_pred.array.probabilities)
    argmax_per_vertex = np.argmax(probs, axis=-1)
    y = np.asarray(y_true)[..., None]
    return float(np.mean(np.any(argmax_per_vertex == y, axis=-1)))


@efficiency.register(ArrayDiscreteCredalSet)
def _efficiency_array_discrete(y_pred: ArrayDiscreteCredalSet) -> float:
    """Average number of distinct argmax classes across the vertex set."""
    probs = np.asarray(y_pred.array.probabilities)
    num_classes = probs.shape[-1]
    argmax_per_vertex = np.argmax(probs, axis=-1)
    classes_picked = (argmax_per_vertex[..., None] == np.arange(num_classes)).any(axis=-2)
    return float(np.mean(classes_picked.sum(axis=-1)))


@coverage.register(ArrayConvexCredalSet)
def _coverage_array_convex(y_pred: ArrayConvexCredalSet, y_true: np.ndarray) -> float:
    """Interval-dominance coverage for a convex credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayConvexCredalSet)
def _efficiency_array_convex(y_pred: ArrayConvexCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a convex credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@coverage.register(ArrayDistanceBasedCredalSet)
def _coverage_array_distance(y_pred: ArrayDistanceBasedCredalSet, y_true: np.ndarray) -> float:
    """Interval-dominance coverage for a distance-based credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayDistanceBasedCredalSet)
def _efficiency_array_distance(y_pred: ArrayDistanceBasedCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a distance-based credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@coverage.register(ArrayProbabilityIntervalsCredalSet)
def _coverage_array_probability_intervals(y_pred: ArrayProbabilityIntervalsCredalSet, y_true: np.ndarray) -> float:
    """Interval-dominance coverage for a probability-intervals credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(ArrayProbabilityIntervalsCredalSet)
def _efficiency_array_probability_intervals(y_pred: ArrayProbabilityIntervalsCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a probability-intervals credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@average_interval_width.register(ArrayDistanceBasedCredalSet)
def _average_interval_width_array_distance(y_pred: ArrayDistanceBasedCredalSet) -> float:
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(ArrayProbabilityIntervalsCredalSet)
def _average_interval_width_array_probability_intervals(y_pred: ArrayProbabilityIntervalsCredalSet) -> float:
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())
