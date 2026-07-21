# experiments/rl_uncertainty/viz/trajectories.py
"""Trajectory overlay on 2D environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import patches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_MEMBER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


def _draw_env(
    ax: Axes,
    obstacles: list[tuple[np.ndarray, float]] | None = None,
    goal: np.ndarray | None = None,
    goal_radius: float = 0.05,
    start: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Draw obstacles, goal, and start markers."""
    if obstacles:
        for center, radius in obstacles:
            circle = patches.Circle((float(center[0]), float(center[1])), radius, color="#546e7a", alpha=0.8, zorder=2)
            ax.add_patch(circle)

    if goal is not None:
        circle = patches.Circle((float(goal[0]), float(goal[1])), goal_radius, color="#4caf50", alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(
            goal[0],
            goal[1],
            "G",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            zorder=3,
        )

    if start is not None:
        circle = patches.Circle((float(start[0]), float(start[1])), 0.02, color="#2196f3", alpha=0.8, zorder=2)
        ax.add_patch(circle)
        ax.text(
            start[0],
            start[1],
            "S",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
            zorder=3,
        )

    if bounds is not None:
        ax.set_xlim(float(bounds[0][0]), float(bounds[1][0]))
        ax.set_ylim(float(bounds[0][1]), float(bounds[1][1]))
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


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
    _draw_env(ax, obstacles, goal, goal_radius, start)

    for traj in risky_trajs:
        pts = np.array(traj["states"])
        ax.plot(pts[:, 0], pts[:, 1], color="#f44336", alpha=0.3, linewidth=0.8, linestyle="--", zorder=1)

    for traj in safe_trajs:
        pts = np.array(traj["states"])
        ax.plot(pts[:, 0], pts[:, 1], color="#2196f3", alpha=0.4, linewidth=1.0, zorder=1)

    n_crash_risky = sum(1 for t in risky_trajs if t["event"] in ("collision", "wall"))
    n_crash_safe = sum(1 for t in safe_trajs if t["event"] in ("collision", "wall"))
    ax.text(
        0.5,
        -0.08,
        f"Risk-neutral crashes: {n_crash_risky}/{len(risky_trajs)}  |  "
        f"Risk-averse crashes: {n_crash_safe}/{len(safe_trajs)}",
        transform=ax.transAxes,
        ha="center",
        fontsize=6,
        color="#666",
    )

    ax.set_title(title, fontsize=9, fontweight="bold")


def plot_member_trajectories(
    ax: Axes,
    member_trajs: list[dict],
    obstacles: list[tuple[np.ndarray, float]] | None = None,
    goal: np.ndarray | None = None,
    goal_radius: float = 0.05,
    start: np.ndarray | None = None,
    title: str = "",
) -> None:
    """Plot individual ensemble member trajectories (spaghetti plot).

    Each member's greedy policy produces a different trajectory,
    and the spread visualizes epistemic uncertainty in action space.

    Args:
        ax: Matplotlib axes.
        member_trajs: List of trajectory dicts, one per ensemble member.
        obstacles: List of (center, radius) obstacle circles.
        goal: Goal position.
        goal_radius: Goal radius.
        start: Start position.
        title: Panel title.
    """
    _draw_env(ax, obstacles, goal, goal_radius, start)

    for i, traj in enumerate(member_trajs):
        pts = np.array(traj["states"])
        color = _MEMBER_COLORS[i % len(_MEMBER_COLORS)]
        event = traj["event"]
        marker = "x" if event in ("collision", "wall") else ("*" if event == "goal" else "")
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.7, linewidth=1.2, label=f"Member {i}", zorder=1)
        if marker:
            ax.plot(pts[-1, 0], pts[-1, 1], marker=marker, color=color, markersize=6, zorder=3)

    ax.legend(fontsize=5, loc="lower right", ncol=2)
    ax.set_title(title, fontsize=9, fontweight="bold")
