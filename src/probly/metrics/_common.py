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


@flexdispatch
def empirical_coverage_classification[T](y_pred: T, y_true: T) -> float:
    """Calculate the empirical coverage for classification."""
    msg = f"Empirical coverage for classification is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@flexdispatch
def empirical_coverage_regression[T](y_pred: T, y_true: T) -> float:
    """Calculate the empirical coverage for regression."""
    msg = f"Empirical coverage for regression is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@flexdispatch
def average_set_size[T](y_pred: T) -> float:
    """Calculate the average prediction set size for classification."""
    msg = f"Average set size for classification is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@flexdispatch
def average_interval_size[T](y_pred: T) -> float:
    """Calculate the average interval size for regression."""
    msg = f"Average interval size for regression is not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)
