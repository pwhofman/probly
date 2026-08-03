# experiments/rl_uncertainty/viz/decomposition.py
"""Aleatoric/epistemic decomposition over training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_decomposition(
    ax: Axes,
    steps: np.ndarray,
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    title: str = "Uncertainty Decomposition",
    window: int = 50,
) -> None:
    """Stacked area plot of uncertainty components over training.

    Args:
        ax: Matplotlib axes.
        steps: Training step indices.
        epistemic: Epistemic uncertainty values per step.
        aleatoric: Aleatoric uncertainty values per step.
        title: Panel title.
        window: Smoothing window size.
    """
    effective_window = min(window, max(len(epistemic) // 2, 1))

    def smooth(arr: np.ndarray) -> np.ndarray:
        if effective_window <= 1:
            return arr
        kernel = np.ones(effective_window) / effective_window
        return np.convolve(arr, kernel, mode="valid")

    s_epi = smooth(epistemic)
    s_alea = smooth(aleatoric)
    s_steps = steps[: len(s_epi)]

    ax.fill_between(s_steps, 0, s_alea, alpha=0.4, color="#2196f3", label="Aleatoric")
    ax.fill_between(s_steps, s_alea, s_alea + s_epi, alpha=0.4, color="#ff9800", label="Epistemic")
    ax.plot(s_steps, s_alea, color="#1565c0", linewidth=0.8)
    ax.plot(s_steps, s_alea + s_epi, color="#e65100", linewidth=0.8)

    ax.set_xlabel("Training Step", fontsize=7)
    ax.set_ylabel("Uncertainty", fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=6)
