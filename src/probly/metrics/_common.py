"""Dispatched metric factories for classification evaluation."""

from __future__ import annotations

from flextype import flexdispatch


@flexdispatch
def auc(x: object, y: object) -> object:
    """Compute area under a curve using the trapezoid rule.

    Args:
        x: x-coordinates (must be monotonic).
        y: y-coordinates.

    Returns:
        Area under the curve.
    """
    msg = f"No auc implementation registered for type {type(x)}"
    raise NotImplementedError(msg)


@flexdispatch
def roc_curve(y_true: object, y_score: object) -> tuple[object, object, object]:
    """Compute receiver operating characteristic (ROC) curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores (higher means more likely positive).

    Returns:
        fpr: False positive rates.
        tpr: True positive rates.
        thresholds: Decreasing score thresholds.
    """
    msg = f"No roc_curve implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


@flexdispatch
def precision_recall_curve(y_true: object, y_score: object) -> tuple[object, object, object]:
    """Compute precision-recall curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores (higher means more likely positive).

    Returns:
        precision: Precision values.
        recall: Recall values.
        thresholds: Decreasing score thresholds.
    """
    msg = f"No precision_recall_curve implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


@flexdispatch
def roc_auc_score(y_true: object, y_score: object) -> object:
    """Compute area under the ROC curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores.

    Returns:
        AUROC value.
    """
    msg = f"No roc_auc_score implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


@flexdispatch
def average_precision_score(y_true: object, y_score: object) -> object:
    """Compute average precision (area under the precision-recall curve).

    Uses the step-function interpolation.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores.

    Returns:
        Average precision value.
    """
    msg = f"No average_precision_score implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


# --- Predicted-set metrics ----------------------------------------------------
#
# Two top-level dispatched functions evaluate predicted-set representations.
# Only four representation types are currently registered; everything else
# raises ``NotImplementedError`` and is intentionally out of scope:
#
# * ``ArrayOneHotConformalSet`` / ``TorchOneHotConformalSet``:
#   classification conformal set. ``y_true`` is integer class labels.
# * ``ArrayIntervalConformalSet`` / ``TorchIntervalConformalSet``:
#   regression conformal set. ``y_true`` is scalar regression targets.
# * ``ArrayConvexCredalSet`` / ``TorchConvexCredalSet``: ``y_true`` is a
#   wrapped ``CategoricalDistribution``; coverage is convex-hull membership
#   (LP feasibility) of the target inside the credal set's vertex hull.
# * ``ArrayProbabilityIntervalsCredalSet`` /
#   ``TorchProbabilityIntervalsCredalSet``: ``y_true`` is a wrapped
#   ``CategoricalDistribution``; coverage is the element-wise
#   ``lower <= target <= upper`` check across all classes.


@flexdispatch
def coverage[T](y_pred: T, y_true: object) -> float:
    """Compute the empirical coverage of a predicted set against ground truth.

    Coverage is the fraction of samples whose true value lies inside the
    predicted set. The expected ``y_true`` type and the membership rule
    depend on the dispatched ``y_pred`` representation:

    * ``OneHotConformalSet``: ``y_true`` is integer class labels of shape
      ``(N,)``. Covered iff the true label is in the conformal set.
    * ``IntervalConformalSet``: ``y_true`` is scalar regression targets of
      shape ``(N,)``. Covered iff ``lower <= y_true <= upper``.
    * ``ConvexCredalSet``: ``y_true`` is a wrapped categorical distribution
      of shape ``(N, K)``. Covered iff the target distribution lies inside
      the convex hull of the credal set's vertex distributions, tested by
      LP feasibility per instance.
    * ``ProbabilityIntervalsCredalSet``: ``y_true`` is a wrapped categorical
      distribution. Covered iff every class satisfies
      ``lower[k] <= target[k] <= upper[k]``.

    Args:
        y_pred: A predicted-set representation.
        y_true: Ground truth aligned with ``y_pred``. Type depends on
            ``y_pred`` (see above).

    Returns:
        Empirical coverage as a float in ``[0, 1]``.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"coverage is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def efficiency[T](y_pred: T) -> float:
    """Compute the average size of a predicted set.

    Smaller is better. Per-type semantics:

    * ``OneHotConformalSet``: average cardinality of the selected class set.
    * ``IntervalConformalSet``: average interval width ``upper - lower``.
    * ``ConvexCredalSet`` / ``ProbabilityIntervalsCredalSet``: cardinality
      of the interval-dominance prediction set built from the credal set's
      ``lower()`` / ``upper()`` envelopes.

    Args:
        y_pred: A predicted-set representation.

    Returns:
        The mean set size as a float.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"efficiency is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)
