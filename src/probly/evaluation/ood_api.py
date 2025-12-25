"""Unified OOD evaluation API for probly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from probly.evaluation.tasks import (
    out_of_distribution_detection_aupr,
    out_of_distribution_detection_auroc,
    out_of_distribution_detection_fnr_at_x_tpr,
    out_of_distribution_detection_fpr_at_x_tpr,
)
from probly.evaluation.types import (
    OodEvaluationResult,
)
from probly.plot.ood import plot_histogram, plot_pr_curve, plot_roc_curve

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure


STATIC_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": out_of_distribution_detection_auroc,
    "aupr": out_of_distribution_detection_aupr,
}

DYNAMIC_METRICS: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
    "fpr": lambda a, b, thr: out_of_distribution_detection_fpr_at_x_tpr(a, b, thr),
    "fnr": lambda a, b, thr: out_of_distribution_detection_fnr_at_x_tpr(a, b, thr),
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

    Examples:
        fpr@0.8
        fnr@95%
        fpr -> default threshold is 0.95
        fnr -> default threshold is 0.95
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
    metrics: None | str | list[str] = None,
) -> float | dict[str, float]:
    """Unified OOD evaluation API.

    Provides backward compatibility while supporting multiple metrics.

    Parameters:
    in_distribution :
        Scores for in-distribution samples.
    out_distribution :
        Scores for out-of-distribution samples.
    metrics : str, list of str, or None
        - None or "auroc": Returns single AUROC value (backward compatible)
        - "all": Returns dict with all available metrics
        - list: Returns dict with specified metrics

    Returns:
    float or dict
        - If metrics is None or "auroc": returns single AUROC float
        - Otherwise: returns dict with metric names as keys
    """
    in_s = np.asarray(in_distribution)
    out_s = np.asarray(out_distribution)

    if metrics is None or metrics == "auroc":
        return STATIC_METRICS["auroc"](in_s, out_s)

    if isinstance(metrics, str):
        metric_list = [*list(STATIC_METRICS.keys()), "fpr", "fnr"] if metrics == "all" else [metrics]
    else:
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


def visualize_ood(
    in_distribution: np.ndarray | list[float],
    out_distribution: np.ndarray | list[float],
    invert_scores: bool = True,
    plot_types: list[str] | None = None,
) -> dict[str, Figure]:
    """Generate visualization plots for OOD evaluation.

    Calculates curve data (ROC, PR) and generates standard plots.

    Parameters:
    in_distribution :
        Scores for in-distribution samples.
    out_distribution :
        Scores for out-of-distribution samples.
    invert_scores : bool
        If True (default), assumes scores are 'Confidence' (High = ID).
        They will be inverted (1.0 - score) for metrics where OOD is the positive class.
        If False, assumes scores are 'Anomaly Scores' (High = OOD).
    plot_types : list[str], optional
        List of specific plots to return (e.g. ['roc', 'hist', 'pr']).
        If None, all plots are generated.

    Returns:
        Dict containing matplotlib Figures for the requested plots.
    """
    available_plots = {"hist", "roc", "pr"}

    requested_plots = available_plots if plot_types is None else set(plot_types).intersection(available_plots)

    id_s = np.asarray(in_distribution)
    ood_s = np.asarray(out_distribution)

    # 1. Calculate scalar metrics using the Unified API
    # -> We use the existing API to ensure consistency in values
    scalars = evaluate_ood(id_s, ood_s, metrics=["auroc", "aupr", "fpr@95tpr"])
    if not isinstance(scalars, dict):
        # Fallback should technically not happen with explicit list input
        scalars = {"auroc": scalars, "aupr": 0.0, "fpr@95tpr": 0.0}

    # 2. Prepare Data for Curves (Requires sklearn)

    fpr, tpr = None, None
    precision, recall = None, None

    if "roc" in requested_plots or "pr" in requested_plots:
        labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
        all_scores = np.concatenate([id_s, ood_s])

        preds = 1.0 - all_scores if invert_scores else all_scores

        # Compute curve arrays conditionally
        if "roc" in requested_plots:
            fpr, tpr, _ = roc_curve(labels, preds)

        if "pr" in requested_plots:
            precision, recall, _ = precision_recall_curve(labels, preds)

    # 3. -> Result Object from types.
    result_data = OodEvaluationResult(
        auroc=scalars["auroc"],
        aupr=scalars["aupr"],
        fpr95=scalars.get("fpr@95tpr"),
        fpr=fpr,
        tpr=tpr,
        precision=precision,
        recall=recall,
        id_scores=id_s,
        ood_scores=ood_s,
    )

    # 4.Generate Plots as figures.
    figures = {}

    if "hist" in requested_plots:
        figures["hist"] = plot_histogram(result_data)

    if "roc" in requested_plots:
        figures["roc"] = plot_roc_curve(result_data)

    if "pr" in requested_plots:
        figures["pr"] = plot_pr_curve(result_data)

    return figures
