# experiments/rl_uncertainty/viz/trajectories.py
"""Trajectory overlay on 2D environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import patches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_trajectories(
    ax: Axes,
    risky_trajs: list[dict],
    safe_trajs: list[dict],
    obstacles: list[tuple[np.ndarray, float]] | None = None,
    goal: np.ndarray | None = None,
    goal_radius: float = 0.05,
    start: np.ndarray | None = None,
    title: str = "",
) -> None:
    """Plot risk-neutral vs risk-averse trajectories on environment.

    Args:
        ax: Matplotlib axes.
        risky_trajs: List of trajectory dicts with 'states' key (risk-neutral).
        safe_trajs: List of trajectory dicts with 'states' key (risk-averse).
        obstacles: List of (center, radius) obstacle circles.
        goal: Goal position.
        goal_radius: Goal radius.
        start: Start position.
        title: Panel title.
    """
    if obstacles:
        for center, radius in obstacles:
            circle = patches.Circle(center, radius, color="#546e7a", alpha=0.8, zorder=2)
            ax.add_patch(circle)

    if goal is not None:
        circle = patches.Circle(goal, goal_radius, color="#4caf50", alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(
            goal[0], goal[1], "G",
            ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3,
        )

    if start is not None:
        circle = patches.Circle(start, 0.02, color="#2196f3", alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(
            start[0], start[1], "S",
            ha="center", va="center", fontsize=6, color="white", fontweight="bold", zorder=3,
        )

    for traj in risky_trajs:
        pts = np.array(traj["states"])
        ax.plot(pts[:, 0], pts[:, 1], color="#f44336", alpha=0.3, linewidth=0.8, linestyle="--", zorder=1)

    for traj in safe_trajs:
        pts = np.array(traj["states"])
        ax.plot(pts[:, 0], pts[:, 1], color="#2196f3", alpha=0.4, linewidth=1.0, zorder=1)

    n_crash_risky = sum(1 for t in risky_trajs if t["event"] in ("collision", "wall"))
    n_crash_safe = sum(1 for t in safe_trajs if t["event"] in ("collision", "wall"))
    ax.text(
        0.5, -0.08,
        f"Risk-neutral crashes: {n_crash_risky}/{len(risky_trajs)}  |  "
        f"Risk-averse crashes: {n_crash_safe}/{len(safe_trajs)}",
        transform=ax.transAxes, ha="center", fontsize=6, color="#666",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)
