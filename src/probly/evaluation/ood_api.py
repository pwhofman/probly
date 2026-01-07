"""Unified OOD evaluation API for probly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sklearn.metrics as sm
from sklearn.metrics import precision_recall_curve, roc_curve

from probly.evaluation.types import (
    OodEvaluationResult,
)
from probly.plot.ood import plot_histogram, plot_pr_curve, plot_roc_curve

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure


def out_of_distribution_detection_auroc(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using prediction functionals from id and ood data.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals
    Returns:
        auroc: float, area under the roc curve

    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    auroc = sm.roc_auc_score(labels, preds)
    return float(auroc)


def out_of_distribution_detection_aupr(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using AUPR (Area Under the Precision-Recall Curve).

    This metric evaluates how well the model distinguishes between in- and out-of-distribution samples,
    focusing more on positive class (OOD) precision and recall.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals

    Returns:
        aupr: float, area under the precision-recall curve
    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    aupr = sm.average_precision_score(labels, preds)
    return float(aupr)


def out_of_distribution_detection_fpr_at_x_tpr(
    in_distribution: np.ndarray,
    out_distribution: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Perform out-of-distribution detection using false positive rate (FPR) at a given true positive rate.

    If no thresholds are specified, the default tpr_target is 0.95.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: numpy.ndarray, scores for in-distribution samples
        out_distribution: numpy.ndarray, scores for out-of-distribution samples
        tpr_target: target TPR value in [0, 1], e.g. 0.95

    Returns:
        fpr_at_target: float, FPR at the first threshold where TPR >= tpr_target

    Notes:
        - Assumes that larger scores correspond to the positive class
          (out-of-distribution).
        - If tpr_target cannot be reached, a ValueError is raised.
    """
    if not 0.0 < tpr_target <= 1.0:
        msg = f"tpr_target must be in the interval (0, 1], got {tpr_target}."
        raise ValueError(msg)

    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate(
        (np.zeros(len(in_distribution)), np.ones(len(out_distribution))),
    )

    fpr, tpr, _ = sm.roc_curve(labels, preds)

    idxs = np.where(tpr >= tpr_target)[0]
    if len(idxs) == 0:
        msg = f"Could not achieve TPR >= {tpr_target:.3f} with given scores."
        raise ValueError(msg)

    first_idx = idxs[0]
    fpr_at_target = fpr[first_idx]
    return float(fpr_at_target)


def out_of_distribution_detection_fnr_at_x_tpr(
    in_distribution: np.ndarray,
    out_distribution: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Perform out-of-distribution detection using false negative rate at user given true positive rate.

    If no thresholds are specified, the default tpr_target is 0.95.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals
        tpr_target: target TPR value in [0, 1], e.g. 0.95

    Returns:
        fnr@X: float, FNR at the first threshold where TPR >= tpr_target
    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    _, tpr, _ = sm.roc_curve(labels, preds)
    idx = np.where(tpr >= tpr_target)[0]
    return float(1.0 - tpr[idx[0]]) if len(idx) else 1.0


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
    metrics: str | list[str] | None = None,
) -> dict[str, float]:
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
    dict[str, float]
        Dictionary mapping metric names to values.
        If metrics is None or "auroc",
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


def compute_ood_evaluation_result(
    in_distribution: np.ndarray | list[float],
    out_distribution: np.ndarray | list[float],
    invert_scores: bool = True,
) -> OodEvaluationResult:
    """Compute all OOD metrics and curve data for visualization.

    Calculation Logic is seperated from the actual plotting logic in the visualize_ood() function.

    Parameters:
    in_distribution :
        Scores for in-distribution samples.
    out_distribution :
        Scores for out-of-distribution samples.
    invert_scores : bool
        If True (default), assumes scores are 'Confidence' (High = ID).
        They will be inverted (1.0 - score) for metrics where OOD is the positive class.
        If False, assumes scores are 'Anomaly Scores' (High = OOD).

    Returns:
        OodEvaluationResult object containing scalars and curve arrays.
    """
    id_s = np.asarray(in_distribution)
    ood_s = np.asarray(out_distribution)

    # 1. Decoupled calculation
    scalars = evaluate_ood(id_s, ood_s, metrics=["auroc", "aupr", "fpr@95tpr"])
    if not isinstance(scalars, dict):
        scalars = {"auroc": scalars, "aupr": 0.0, "fpr@95tpr": 0.0}

    # 2. Prepare Data for Curve
    labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    all_scores = np.concatenate([id_s, ood_s])

    preds = 1.0 - all_scores if invert_scores else all_scores

    fpr, tpr, _ = roc_curve(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)

    # 3. Return Result Object
    return OodEvaluationResult(
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


def visualize_ood(
    result_data: OodEvaluationResult,
    plot_types: list[str] | None = None,
) -> dict[str, Figure]:
    """Generate visualization plots from OODEvaluationResult type.

    Parameters:
    results : OodEvaluationResult
        The calculated result object containing scores and curve data.
        Use `compute_ood_evaluation_result` to generate this.
    plot_types : list[str], optional
        List of specific plots to return (e.g. ['roc', 'hist', 'pr']).
        If None, all plots are generated.

    Returns:
        Dict containing matplotlib Figures for the requested plots.
    """
    available_plots = {"hist", "roc", "pr"}
    requested_plots = available_plots if plot_types is None else set(plot_types).intersection(available_plots)
    figures = {}

    if "hist" in requested_plots:
        figures["hist"] = plot_histogram(result_data)

    if "roc" in requested_plots:
        figures["roc"] = plot_roc_curve(result_data)

    if "pr" in requested_plots:
        figures["pr"] = plot_pr_curve(result_data)

    return figures
