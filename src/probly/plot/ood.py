"""Plotting utilities for OOD evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from probly.evaluation.types import OodEvaluationResult


def plot_histogram(
    results: OodEvaluationResult,
    ax: Axes | None = None,
    bins: int = 50,
    title: str = "Score Distribution",
) -> Figure:
    """Plot ID vs OOD score histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if results.id_scores is None or results.ood_scores is None:
        msg = "Results object must contain raw scores for histogram."
        raise ValueError(msg)

    ax.hist(
        results.id_scores,
        bins=bins,
        alpha=0.6,
        density=True,
        label="In-Distribution",
    )
    ax.hist(
        results.ood_scores,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Out-of-Distribution",
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def plot_roc_curve(
    results: OodEvaluationResult,
    ax: Axes | None = None,
) -> Figure:
    """Plot ROC curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if results.fpr is None or results.tpr is None:
        msg = "Results object missing FPR/TPR arrays."
        raise ValueError(msg)

    label = f"AUROC = {results.auroc:.3f}"
    if results.fpr95 is not None:
        label += f" (FPR@95 = {results.fpr95:.3f})"

    ax.plot(results.fpr, results.tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def plot_pr_curve(
    results: OodEvaluationResult,
    ax: Axes | None = None,
) -> Figure:
    """Plot Precision-Recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if results.precision is None or results.recall is None:
        msg = "Results object missing Precision/Recall arrays."
        raise ValueError(msg)

    ax.plot(
        results.recall,
        results.precision,
        lw=2,
        color="green",
        label=f"AUPR = {results.aupr:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig
