"""Plotting for Coverage and Efficiency metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import probly.visualization.config as cfg

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class CoverageEfficiencyVisualizer:
    """Class to visualize the trade-off between coverage and set size (efficiency)."""

    def __init__(self) -> None:
        """Initialize the visualizer."""

    def _compute_metrics(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        alphas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute empirical coverage and mean set size for various alpha thresholds."""
        n_samples, _ = probs.shape

        sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        coverages = []
        efficiencies = []

        for alpha in alphas:
            confidence_level = 1.0 - alpha

            cutoffs = np.argmax(cumsum_probs >= confidence_level, axis=1)
            set_sizes = cutoffs + 1

            target_ranks = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                target_ranks[i] = np.where(sorted_indices[i] == targets[i])[0][0]

            is_covered = target_ranks <= cutoffs

            coverages.append(np.mean(is_covered))
            efficiencies.append(np.mean(set_sizes))

        return np.array(coverages), np.array(efficiencies)

    def plot_coverage_efficiency(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        title: str = "Coverage vs. Efficiency",
        ax: Axes | None = None,
    ) -> Axes:
        """Create a dual-axis plot showing Coverage and Efficiency over Confidence Levels.

        Args:
            probs: Array of predicted probabilities with shape (n_samples, n_classes).
            targets: True labels as a 1D array of shape (n_samples,).
            title: Plot title. Defaults to "Coverage vs. Efficiency".
            ax: Matplotlib Axes to draw the plot on. If None, a new figure and axes are created.

        Returns:
            Matplotlib Axes containing the coverage-efficiency plot.
        """
        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 5))

        alphas = np.linspace(0.001, 0.999, 100)
        confidence_levels = 1.0 - alphas

        coverages, efficiencies = self._compute_metrics(probs, targets, alphas)
        n_classes = probs.shape[1]
        efficiency_norm = 1.0 - (efficiencies - 1.0) / (n_classes - 1.0)

        ax.plot(
            confidence_levels,
            confidence_levels,
            color=cfg.LINES,
            linestyle=cfg.MIN_MAX_LINESTYLE_1,
            label="Ideal Coverage",
            zorder=0,
        )

        line1 = ax.plot(
            confidence_levels,
            coverages,
            color=cfg.BLUE,
            linewidth=cfg.HULL_LINE_WIDTH,
            label="Empirical Coverage",
            zorder=2,
        )

        ax.set_xlabel(r"Target Confidence Level ($1 - \alpha$)", fontsize=cfg.PROB_FONT_SIZE)
        ax.set_ylabel("Empirical Coverage", color=cfg.BLUE, fontsize=cfg.PROB_FONT_SIZE)

        ax.tick_params(axis="y", labelcolor=cfg.BLUE, labelsize=cfg.PROB_FONT_SIZE)
        ax.tick_params(axis="x", labelsize=cfg.PROB_FONT_SIZE)

        ax.set_ylim(1.05, -0.05)

        ax2 = ax.twinx()
        line2 = ax2.plot(
            confidence_levels,
            efficiency_norm,
            color=cfg.RED,
            linewidth=cfg.HULL_LINE_WIDTH,
            linestyle=cfg.MIN_MAX_LINESTYLE_2,
            label="Efficiency (1 - Norm. Set Size)",
            zorder=2,
        )

        ax2.set_ylabel("Efficiency", color=cfg.RED, fontsize=cfg.PROB_FONT_SIZE)
        ax2.tick_params(axis="y", labelcolor=cfg.RED, labelsize=cfg.PROB_FONT_SIZE)

        ax2.set_ylim(-0.05, 1.05)

        lines = line1 + line2
        labels_legend: list[str] = [str(li.get_label()) for li in lines]

        ax.legend(lines, labels_legend, loc="upper right", fontsize=cfg.PROB_FONT_SIZE)

        ax.set_title(title, pad=15, fontsize=cfg.PROB_FONT_SIZE + 2)

        valid_styles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        raw_style = getattr(cfg, "PROB_LINESTYLE", ":")
        grid_style = raw_style if raw_style in valid_styles else ":"

        ax.grid(True, linestyle=grid_style, color=cfg.LINES, alpha=cfg.FILL_ALPHA)

        return ax
