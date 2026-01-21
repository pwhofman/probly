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

        # Sort probabilities descending for cumulative sum strategy
        sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        coverages = []
        efficiencies = []

        for alpha in alphas:
            confidence_level = 1.0 - alpha

            # Determine set size: include classes until cumulative prob >= confidence_level
            cutoffs = np.argmax(cumsum_probs >= confidence_level, axis=1)
            set_sizes = cutoffs + 1

            # Check if true target is in the set
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
        """Create a dual-axis plot showing Coverage and Efficiency over Confidence Levels."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        # Define range of confidence levels
        alphas = np.linspace(0.001, 0.999, 100)
        confidence_levels = 1.0 - alphas

        # Compute metrics
        coverages, efficiencies = self._compute_metrics(probs, targets, alphas)
        n_classes = probs.shape[1]

        # Normalized set size: 1.0 = Best (size 1), 0.0 = Worst (size N)
        efficiency_norm = 1.0 - (efficiencies - 1.0) / (n_classes - 1.0)

        # --- Plot Coverage (Primary Y-Axis) ---
        ax.plot(
            confidence_levels,
            confidence_levels,
            color=cfg.LINES,
            linestyle="--",
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

        ax.set_xlabel(r"Target Confidence Level ($1 - \alpha$)")
        ax.set_ylabel("Empirical Coverage", color=cfg.BLUE)
        ax.tick_params(axis="y", labelcolor=cfg.BLUE)

        # Coverage Axis inverted: 0 at top, 1 at bottom
        ax.set_ylim(1.05, -0.05)

        # --- Plot Efficiency (Secondary Y-Axis) ---
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

        ax2.set_ylabel("Efficiency", color=cfg.RED)
        ax2.tick_params(axis="y", labelcolor=cfg.RED)

        # Efficiency Axis Normal: 1 at top, 0 at bottom
        ax2.set_ylim(-0.05, 1.05)

        # --- Combine Legends ---
        lines = line1 + line2
        labels_legend = [l.get_label() for l in lines]  # noqa: E741

        ax.legend(lines, labels_legend, loc="upper right")

        ax.set_title(title, pad=15)

        # --- FIX: Safe Grid Style ---
        valid_styles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        raw_style = getattr(cfg, "PROB_LINESTYLE", ":")
        grid_style = raw_style if raw_style in valid_styles else ":"

        ax.grid(True, linestyle=grid_style, alpha=0.3)

        return ax
