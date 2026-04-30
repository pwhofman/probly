# experiments/rl_uncertainty/viz/triptych.py
"""Compose the 3-panel triptych figure."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from .decomposition import plot_decomposition
from .heatmap import plot_eu_heatmap
from .trajectories import plot_member_trajectories, plot_trajectories


def make_triptych(
    eval_dir: Path,
    obstacles: list[tuple[np.ndarray, float]] | None = None,
    goal: np.ndarray | None = None,
    goal_radius: float = 0.05,
    start: np.ndarray | None = None,
    train_log_dir: Path | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (11, 3.5),
) -> plt.Figure:
    """Create the 3-panel triptych from evaluation results.

    Args:
        eval_dir: Directory containing trajectories_risky.json,
            trajectories_safe.json, eu_grid.json.
        obstacles: Environment obstacles for overlay.
        goal: Goal position.
        goal_radius: Goal hit radius.
        start: Start position.
        train_log_dir: Directory with uncertainty_log.json for decomposition panel.
        output_path: If set, save figure to this path.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    risky_trajs = json.loads((eval_dir / "trajectories_risky.json").read_text())
    safe_trajs = json.loads((eval_dir / "trajectories_safe.json").read_text())
    eu_grid_data = json.loads((eval_dir / "eu_grid.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Panel (a): Trajectories — prefer ensemble member spaghetti plot if available
    member_path = eval_dir / "trajectories_members.json"
    if member_path.exists():
        member_trajs = json.loads(member_path.read_text())
        plot_member_trajectories(
            axes[0], member_trajs,
            obstacles=obstacles, goal=goal, goal_radius=goal_radius, start=start,
            title="(a) Ensemble Member Trajectories",
        )
    else:
        plot_trajectories(
            axes[0], risky_trajs, safe_trajs,
            obstacles=obstacles, goal=goal, goal_radius=goal_radius, start=start,
            title="(a) Trajectories",
        )

    # Panel (b): EU Heatmap
    plot_eu_heatmap(
        axes[1], fig,
        xs=np.array(eu_grid_data["xs"]),
        ys=np.array(eu_grid_data["ys"]),
        eu_grid=np.array(eu_grid_data["epistemic"]),
        obstacles=obstacles,
        safe_trajs=safe_trajs,
        title="(b) Epistemic Uncertainty",
    )

    # Panel (c): Decomposition
    if train_log_dir and (train_log_dir / "uncertainty_log.json").exists():
        unc_log = json.loads((train_log_dir / "uncertainty_log.json").read_text())
        steps = np.array([e["step"] for e in unc_log])
        epi_vals = np.array([e["mean_epistemic"] for e in unc_log])
        alea_vals = np.array([e["mean_aleatoric"] for e in unc_log])
        plot_decomposition(axes[2], steps, epi_vals, alea_vals, title="(c) Decomposition")
    else:
        # Fallback: static decomposition from EU grid (sorted descending)
        epi_vals = np.sort(np.array(eu_grid_data["epistemic"]).ravel())[::-1]
        alea_vals = np.sort(np.array(eu_grid_data["aleatoric"]).ravel())[::-1]
        steps = np.arange(len(epi_vals))
        plot_decomposition(axes[2], steps, epi_vals, alea_vals, title="(c) Uncertainty Distribution")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig
