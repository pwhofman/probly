"""Plotting utilities for OOD evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .config import PlotConfig
from .utils import _plot_line_with_outline

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    import numpy as np


def _resolve_axes(ax: Axes | None, config: PlotConfig) -> tuple[Figure | SubFigure, Axes]:
    """Create or retrieve Figure and Axes.

    If *ax* is ``None`` a new figure is created using *config*.  Otherwise the
    figure is retrieved from the provided axes.

    Args:
        ax: Optional pre-existing axes for subplot composition.
        config: Plot configuration for figure defaults.

    Returns:
        A (figure, axes) tuple.

    Raises:
        RuntimeError: If the axes has no associated figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.dpi)
    else:
        fig = ax.get_figure()
    if fig is None:
        msg = "Could not get figure from axes."
        raise RuntimeError(msg)
    return fig, ax


def plot_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    ax: Axes | None = None,
    bins: int = 50,
    title: str = "Score Distribution",
    config: PlotConfig | None = None,
) -> Figure | SubFigure:
    """Plot a density histogram comparing ID and OOD score distributions.

    Two overlapping histograms are drawn — one for in-distribution scores and
    one for out-of-distribution scores — so the degree of overlap (and
    therefore the difficulty of the detection task) is immediately visible.

    Args:
        id_scores: Uncertainty or anomaly scores for in-distribution samples.
        ood_scores: Uncertainty or anomaly scores for out-of-distribution
            samples.
        ax: Optional pre-existing axes for subplot composition.  When
            ``None`` a new figure is created.
        bins: Number of histogram bins.
        title: Axes title string.
        config: Plot configuration.  Defaults to ``PlotConfig()`` when
            ``None``.

    Returns:
        The figure that contains the histogram.

    Raises:
        RuntimeError: If *ax* is provided but has no associated figure.
    """
    config = config or PlotConfig()
    fig, ax = _resolve_axes(ax, config)

    ax.hist(
        id_scores,
        bins=bins,
        alpha=config.histogram_alpha,
        density=True,
        color=config.color_negative,
        label="In-Distribution",
    )
    ax.hist(
        ood_scores,
        bins=bins,
        alpha=config.histogram_alpha,
        density=True,
        color=config.color_positive,
        label="Out-of-Distribution",
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.grid(True, linestyle=config.grid_linestyle, alpha=config.grid_alpha)

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    fpr95: float | None = None,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
) -> Figure | SubFigure:
    """Plot a Receiver Operating Characteristic (ROC) curve.

    The curve is annotated with the AUROC score and, optionally, the false
    positive rate at 95 % TPR.  A diagonal reference line representing a
    random classifier is included for visual calibration.

    Args:
        fpr: False positive rate values as returned by
            ``sklearn.metrics.roc_curve``.
        tpr: True positive rate values as returned by
            ``sklearn.metrics.roc_curve``.
        auroc: Area under the ROC curve.
        fpr95: False positive rate at 95 % TPR.  When provided it is appended
            to the legend label.
        ax: Optional pre-existing axes for subplot composition.  When
            ``None`` a new figure is created.
        config: Plot configuration.  Defaults to ``PlotConfig()`` when
            ``None``.

    Returns:
        The figure that contains the ROC curve.

    Raises:
        RuntimeError: If *ax* is provided but has no associated figure.
    """
    config = config or PlotConfig()
    fig, ax = _resolve_axes(ax, config)

    label = f"AUROC = {auroc:.3f}"
    if fpr95 is not None:
        label += f" (FPR@95 = {fpr95:.3f})"

    _plot_line_with_outline(
        ax,
        fpr,
        tpr,
        linewidth=config.line_width,
        color=config.color(0),
        label=label,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=config.color_neutral,
        label="Random",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=config.grid_linestyle, alpha=config.grid_alpha)

    return fig


def plot_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    aupr: float,
    ax: Axes | None = None,
    config: PlotConfig | None = None,
) -> Figure | SubFigure:
    """Plot a Precision-Recall (PR) curve.

    The curve is annotated with the AUPR score in the legend.  Higher curves
    indicate better ability to distinguish OOD from ID samples across
    different decision thresholds.

    Args:
        recall: Recall values as returned by
            ``sklearn.metrics.precision_recall_curve``.
        precision: Precision values as returned by
            ``sklearn.metrics.precision_recall_curve``.
        aupr: Area under the precision-recall curve.
        ax: Optional pre-existing axes for subplot composition.  When
            ``None`` a new figure is created.
        config: Plot configuration.  Defaults to ``PlotConfig()`` when
            ``None``.

    Returns:
        The figure that contains the PR curve.

    Raises:
        RuntimeError: If *ax* is provided but has no associated figure.
    """
    config = config or PlotConfig()
    fig, ax = _resolve_axes(ax, config)

    _plot_line_with_outline(
        ax,
        recall,
        precision,
        linewidth=config.line_width,
        color=config.color(0),
        label=f"AUPR = {aupr:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=config.grid_linestyle, alpha=config.grid_alpha)

    return fig
