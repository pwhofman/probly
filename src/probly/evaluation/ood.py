"""Unified OOD evaluation API for probly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sklearn.metrics as sm

from probly.evaluation.tasks import (
    fpr_at_tpr,
    out_of_distribution_detection,
    out_of_distribution_detection_aupr,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# ------------------------------------------------------------
# Registry for static predefined metrics
# ------------------------------------------------------------
STATIC_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": out_of_distribution_detection,
    "aupr": out_of_distribution_detection_aupr,
    "fpr@95": lambda a, b: fpr_at_tpr(a, b, 0.95),
}


# ------------------------------------------------------------
# Dynamic metric dispatch table
# ------------------------------------------------------------
def _compute_fnr_at_tpr(
    in_s: np.ndarray,
    out_s: np.ndarray,
    tpr_target: float,
) -> float:
    """Compute FNR at specified TPR threshold."""
    scores = np.concatenate((in_s, out_s))
    labels = np.concatenate((np.zeros_like(in_s), np.ones_like(out_s)))
    fpr, tpr, _ = sm.roc_curve(labels, scores)
    idx = np.where(tpr >= tpr_target)[0]
    return float(1.0 - tpr[idx[0]]) if len(idx) else 1.0


DYNAMIC_METRICS: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
    "fpr": lambda a, b, thr: fpr_at_tpr(a, b, thr),
    "tnr": lambda a, b, thr: 1.0 - fpr_at_tpr(a, b, thr),
    "fnr": _compute_fnr_at_tpr,
}


# ------------------------------------------------------------
# Utility: parse dynamic metric names
# ------------------------------------------------------------
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

    Examples:
        fpr@0.8
        fnr@95%
        tnr@0.99
    """
    try:
        base, t = spec.split("@")
        base = base.lower().strip()
        t = t.strip().lower()

        threshold = float(t[:-1]) / 100 if t.endswith("%") else float(t)
        _validate_threshold(threshold)
        _validate_base_metric(base)

    except Exception as e:
        msg = f"Invalid metric specification '{spec}'. Use e.g. 'fpr@0.8' or 'fnr@90%'."
        raise ValueError(msg) from e
    else:
        return base, threshold


# ------------------------------------------------------------
# Unified API
# ------------------------------------------------------------
def evaluate_ood(
    in_distribution: np.ndarray | list[float],
    out_distribution: np.ndarray | list[float],
    metrics: None | str | list[str] = None,
) -> float | dict[str, float]:
    """Unified OOD evaluation API.

    Provides backward compatibility while supporting multiple metrics.

    Parameters:
    -----------
    in_distribution : array-like
        Scores for in-distribution samples.
    out_distribution : array-like
        Scores for out-of-distribution samples.
    metrics : str, list of str, or None, optional
        - None or "auroc": Returns single AUROC value (backward compatible)
        - "all": Returns dict with all available metrics
        - list: Returns dict with specified metrics

    Returns:
    --------
    float or dict
        - If metrics is None or "auroc": returns single AUROC float
        - Otherwise: returns dict with metric names as keys
    """
    in_s = np.asarray(in_distribution)
    out_s = np.asarray(out_distribution)

    # backward compatibility -> single AUROC
    if metrics is None or metrics == "auroc":
        return STATIC_METRICS["auroc"](in_s, out_s)

    # resolve metric list
    if isinstance(metrics, str):
        metric_list = list(STATIC_METRICS.keys()) if metrics == "all" else [metrics]
    else:
        metric_list = list(metrics)

    results: dict[str, float] = {}

    for metric_name in metric_list:
        metric_name_lower = metric_name.lower().strip()

        if metric_name_lower in STATIC_METRICS:
            results[metric_name] = STATIC_METRICS[metric_name_lower](in_s, out_s)
            continue

        if "@" in metric_name_lower:
            base, thr = parse_dynamic_metric(metric_name_lower)
            results[metric_name] = DYNAMIC_METRICS[base](in_s, out_s, thr)
            continue

        msg = f"Unknown metric '{metric_name}'. Available: {list(STATIC_METRICS.keys())} + dynamic metric@value."
        raise ValueError(msg)

    return results
