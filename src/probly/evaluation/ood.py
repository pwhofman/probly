"""Unified OOD evaluation API for probly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from probly.metrics import average_precision_score, roc_auc_score, roc_curve

if TYPE_CHECKING:
    from collections.abc import Callable


def out_of_distribution_detection_auroc(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using prediction functionals.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: In-distribution prediction functionals.
        out_distribution: Out-of-distribution prediction functionals.

    Returns:
        Area under the ROC curve.

    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    auroc = roc_auc_score(labels, preds)
    return float(auroc)  # ty:ignore[invalid-argument-type]


def out_of_distribution_detection_aupr(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using AUPR.

    This metric evaluates how well the model distinguishes between in- and out-of-distribution samples,
    focusing more on positive class (OOD) precision and recall.

    Args:
        in_distribution: In-distribution prediction functionals.
        out_distribution: Out-of-distribution prediction functionals.

    Returns:
        Area under the precision-recall curve.

    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    aupr = average_precision_score(labels, preds)
    return float(aupr)  # ty:ignore[invalid-argument-type]


def out_of_distribution_detection_fpr_at_x_tpr(
    in_distribution: np.ndarray,
    out_distribution: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Perform out-of-distribution detection using false positive rate at a given true positive rate.

    If no thresholds are specified, the default tpr_target is 0.95.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: Scores for in-distribution samples.
        out_distribution: Scores for out-of-distribution samples.
        tpr_target: Target TPR value in (0, 1].

    Returns:
        False positive rate at the first threshold where TPR >= tpr_target.

    Raises:
        ValueError: If tpr_target is not in (0, 1] or cannot be achieved.

    Note:
        Assumes that larger scores correspond to the positive class (out-of-distribution).

    """
    if not 0.0 < tpr_target <= 1.0:
        msg = f"tpr_target must be in the interval (0, 1], got {tpr_target}."
        raise ValueError(msg)

    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate(
        (np.zeros(len(in_distribution)), np.ones(len(out_distribution))),
    )

    fpr, tpr, _ = roc_curve(labels, preds)

    idxs = np.where(tpr >= tpr_target)[0]  # ty:ignore[unsupported-operator]
    if len(idxs) == 0:
        msg = f"Could not achieve TPR >= {tpr_target:.3f} with given scores."
        raise ValueError(msg)

    first_idx = idxs[0]
    fpr_at_target = fpr[first_idx]  # ty:ignore[not-subscriptable]
    return float(fpr_at_target)


def out_of_distribution_detection_fnr_at_x_tpr(
    in_distribution: np.ndarray,
    out_distribution: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Perform out-of-distribution detection using false negative rate at a given true positive rate.

    If no thresholds are specified, the default tpr_target is 0.95.

    Args:
        in_distribution: In-distribution prediction functionals.
        out_distribution: Out-of-distribution prediction functionals.
        tpr_target: Target TPR value in (0, 1].

    Returns:
        False negative rate at the first threshold where TPR >= tpr_target.

    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    _, tpr, _ = roc_curve(labels, preds)
    idx = np.where(tpr >= tpr_target)[0]  # ty:ignore[unsupported-operator]
    return float(1.0 - tpr[idx[0]]) if len(idx) else 1.0  # ty:ignore[not-subscriptable]


STATIC_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": out_of_distribution_detection_auroc,
    "aupr": out_of_distribution_detection_aupr,
}

DYNAMIC_METRICS: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
    "fpr": out_of_distribution_detection_fpr_at_x_tpr,
    "fnr": out_of_distribution_detection_fnr_at_x_tpr,
}


def _validate_threshold(threshold: float) -> None:
    """Validate threshold is in (0, 1]."""
    if not 0 < threshold <= 1:
        msg = "threshold must be in (0,1]"
        raise ValueError(msg)


def _validate_base_metric(base: str) -> None:
    """Validate base metric is known."""
    if base not in DYNAMIC_METRICS:
        msg = f"unknown dynamic metric '{base}'"
        raise ValueError(msg)


def parse_dynamic_metric(spec: str) -> tuple[str, float]:
    """Parse dynamic metric specification.

    Args:
        spec: Metric specification string (e.g., 'fpr@0.8', 'fnr@95%', 'fpr').

    Returns:
        A tuple containing:
            - base: The base metric name ('fpr' or 'fnr').
            - threshold: The threshold value. Defaults to 0.95 if not specified.

    Raises:
        ValueError: If specification is invalid.

    Example:
        >>> parse_dynamic_metric('fpr@0.8')
        ('fpr', 0.8)
        >>> parse_dynamic_metric('fnr@95%')
        ('fnr', 0.95)
        >>> parse_dynamic_metric('fpr')
        ('fpr', 0.95)

    """
    try:
        spec = spec.lower().strip()

        if "@" not in spec:
            base = spec
            threshold = 0.95
        else:
            base, t = spec.split("@")
            base = base.lower().strip()
            t = t.strip().lower()

            threshold = float(t[:-1]) / 100 if t.endswith("%") else float(t)

        _validate_threshold(threshold)
        _validate_base_metric(base)

    except Exception as e:
        msg = f"Invalid metric specification '{spec}'. Use e.g. 'fpr', 'fpr@0.8' or 'fnr@90%'."
        raise ValueError(msg) from e
    else:
        return base, threshold


def evaluate_ood(
    in_distribution: np.ndarray | list[float],
    out_distribution: np.ndarray | list[float],
    metrics: str | list[str] | None = None,
) -> dict[str, float]:
    """Unified OOD evaluation API.

    Provides backward compatibility while supporting multiple metrics.

    Args:
        in_distribution: Scores for in-distribution samples.
        out_distribution: Scores for out-of-distribution samples.
        metrics: Metrics to compute. Can be:
            - None or "auroc": Returns single AUROC value (backward compatible).
            - "all": Returns dict with all available metrics.
            - list: Returns dict with specified metrics.

    Returns:
        A dictionary mapping metric names to values. If metrics is None or "auroc",
        the dict contains only the "auroc" entry.

    """
    in_s = np.asarray(in_distribution)
    out_s = np.asarray(out_distribution)

    if metrics is None:
        return {"auroc": STATIC_METRICS["auroc"](in_s, out_s)}

    if isinstance(metrics, str):
        if metrics == "auroc":
            return {"auroc": STATIC_METRICS["auroc"](in_s, out_s)}
        metric_list = [*STATIC_METRICS.keys(), "fpr", "fnr"] if metrics == "all" else [metrics]
    else:
        # metrics is a list list[str]
        metric_list = list(metrics)

    results: dict[str, float] = {}

    for metric_name in metric_list:
        metric_name_lower = metric_name.lower().strip()

        if metric_name_lower in STATIC_METRICS:
            results[metric_name] = STATIC_METRICS[metric_name_lower](in_s, out_s)

        elif metric_name_lower in DYNAMIC_METRICS or "@" in metric_name_lower:
            base, thr = parse_dynamic_metric(metric_name_lower)
            results[metric_name] = DYNAMIC_METRICS[base](in_s, out_s, thr)

        else:
            msg = (
                f"Unknown metric '{metric_name}'. "
                f"Available: {list(STATIC_METRICS.keys())} "
                " + dynamic metrics 'fpr', 'fnr' (optionally with @value)."
            )
            raise ValueError(msg)
    return results
