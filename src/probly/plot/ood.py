"""Visualization Modules for OOD Evaluation in Probly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import numpy as np


def plot_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    ax: Axes | None = None,
    title: str = "Confidence Score Distribution",
) -> Figure:
    """Plot histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.hist(id_scores, bins=30, alpha=0.6, label="In-Distribution (ID)", density=True)
    ax.hist(ood_scores, bins=30, alpha=0.6, label="Out-of-Distribution (OOD)", density=True)

    ax.set_title(title)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    ax: Axes | None = None,
) -> Figure:
    """Plot roc_curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUROC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random Guess")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig
