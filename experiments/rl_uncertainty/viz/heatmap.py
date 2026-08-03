# experiments/rl_uncertainty/viz/heatmap.py
"""Epistemic uncertainty heatmap over state space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import patches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_eu_heatmap(
    ax: Axes,
    fig: Figure,
    xs: np.ndarray,
    ys: np.ndarray,
    eu_grid: np.ndarray,
    obstacles: list[tuple[np.ndarray, float]] | None = None,
    safe_trajs: list[dict] | None = None,
    title: str = "Epistemic Uncertainty",
    cmap: str = "RdYlBu_r",
) -> None:
    """Plot EU heatmap with optional obstacle and trajectory overlays.

    Args:
        ax: Matplotlib axes.
        fig: Figure (for colorbar).
        xs: X grid coordinates.
        ys: Y grid coordinates.
        eu_grid: EU values, shape (len(ys), len(xs)).
        obstacles: Obstacle circles to overlay.
        safe_trajs: Risk-averse trajectories to overlay.
        title: Panel title.
        cmap: Colormap name.
    """
    im = ax.imshow(
        eu_grid,
        origin="lower",
        extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
        cmap=cmap,
        aspect="equal",
        interpolation="bilinear",
    )

    if obstacles:
        for center, radius in obstacles:
            circle = patches.Circle(
                (float(center[0]), float(center[1])),
                radius,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                linestyle="--",
                zorder=2,
            )
            ax.add_patch(circle)

    if safe_trajs:
        for traj in safe_trajs[:5]:
            pts = np.array(traj["states"])
            ax.plot(pts[:, 0], pts[:, 1], color="white", alpha=0.5, linewidth=0.8, zorder=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Epistemic Uncertainty", fontsize=6)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)
