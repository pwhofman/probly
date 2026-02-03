"""Plotting utilities for OOD evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    import numpy as np


def plot_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    ax: Axes | None = None,
    bins: int = 50,
    title: str = "Score Distribution",
) -> Figure | SubFigure:
    """Plot ID vs OOD score histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if fig is None:
        msg = "Could not get figure from axes."
        raise RuntimeError(msg)

    ax.hist(
        id_scores,
        bins=bins,
        alpha=0.6,
        density=True,
        label="In-Distribution",
    )
    ax.hist(
        ood_scores,
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
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    fpr95: float | None = None,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plot ROC curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if fig is None:
        msg = "Could not get figure from axes."
        raise RuntimeError(msg)

    label = f"AUROC = {auroc:.3f}"
    if fpr95 is not None:
        label += f" (FPR@95 = {fpr95:.3f})"

    ax.plot(fpr, tpr, lw=2, label=label)
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
    recall: np.ndarray,
    precision: np.ndarray,
    aupr: float,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plot Precision-Recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if fig is None:
        msg = "Could not get figure from axes."
        raise RuntimeError(msg)

    ax.plot(
        recall,
        precision,
        lw=2,
        color="green",
        label=f"AUPR = {aupr:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig
