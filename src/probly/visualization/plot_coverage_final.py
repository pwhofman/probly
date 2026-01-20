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
        ax.set_ylim(0, 1.05)

        # --- Plot Efficiency (Secondary Y-Axis) ---
        ax2 = ax.twinx()
        line2 = ax2.plot(
            confidence_levels,
            efficiency_norm,
            color=cfg.RED,
            linewidth=cfg.HULL_LINE_WIDTH,
            linestyle=cfg.MIN_MAX_LINESTYLE_2,
            label="Efficiency",
            zorder=2,
        )

        ax2.set_ylabel("Efficiency", color=cfg.RED)
        ax2.tick_params(axis="y", labelcolor=cfg.RED)
        ax2.set_ylim(1.05, -0.05)

        # --- Combine Legends ---
        lines = line1 + line2
        labels_legend = [l.get_label() for l in lines]  # noqa: E741
        ax.legend(lines, labels_legend, loc="lower right")

        ax.set_title(title, pad=15)

        # --- FIX: Safe Grid Style ---
        # Prüft, ob der Style aus der Config gültig ist. Wenn nicht (z.B. '..'), nutze ':'
        valid_styles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        raw_style = getattr(cfg, "PROB_LINESTYLE", ":")
        grid_style = raw_style if raw_style in valid_styles else ":"

        ax.grid(True, linestyle=grid_style, alpha=0.3)

        return ax


# --- DEMO / TEST SECTION ---


def generate_mock_data(n_samples: int = 2000, n_classes: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data."""
    np.random.seed(42)  # noqa: NPY002
    targets = np.random.randint(0, n_classes, size=n_samples)  # noqa: NPY002
    logits = np.random.randn(n_samples, n_classes)  # noqa: NPY002

    scale = np.random.uniform(0.5, 2.5, size=(n_samples, 1))  # noqa: NPY002
    for i in range(n_samples):
        logits[i, targets[i]] += np.random.uniform(2.0, 5.0)  # noqa: NPY002

    logits = logits * scale
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probs, targets


if __name__ == "__main__":
    probs_mock, targets_mock = generate_mock_data(n_samples=2000, n_classes=20)

    viz = CoverageEfficiencyVisualizer()

    # Einzelfenster-Fix:
    fig, ax = plt.subplots(figsize=(10, 6))

    viz.plot_coverage_efficiency(
        probs_mock,
        targets_mock,
        title="Demo: Coverage vs. Efficiency",
        ax=ax,
    )

    plt.tight_layout()
    plt.show()
