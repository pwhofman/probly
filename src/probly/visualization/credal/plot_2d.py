"""Plotting for 2 class probabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import probly.visualization.config as cfg

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class IntervalVisualizer:
    """Class to create 2d plots."""

    def __init__(self) -> None:  # noqa: D107
        pass

    def probs_to_coords_2d(self, probs: np.ndarray) -> tuple:
        """Convert 2D probabilities to 2D coordinates.

        Args:
            probs: Probability vector for 2 classes.

        Returns:
            Cartesian coordinates as tuple.
        """
        _, p2 = probs
        x = p2
        y = 0
        return x, y

    def interval_plot(
        self,
        probs: np.ndarray,
        labels: list[str],
        title: str,
        mle_flag: bool,
        credal_flag: bool,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the interval plot.

        Args:
            probs: Probability vector for 2 classes.
            labels: Labels for the interval plot.
            title: Title of the plot.
            mle_flag: Flag to indicate whether median of probabilities is shown.
            credal_flag: Flag to indicate whether min/max interval is shown.
            ax: Axes to draw the plot on. If None, a new Axes is created.

        Returns:
            Interval plot.
        """
        coords = np.array([self.probs_to_coords_2d(p) for p in probs])

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        y_marg = np.array([0.1, -0.1])

        plt.plot([0, 1], [0, 0], color=cfg.BLACK, linewidth=cfg.HULL_LINE_WIDTH, zorder=0)

        if mle_flag:
            mle = np.mean(coords[:, 0])
            ax.scatter(mle, 0, color=cfg.RED, s=50, zorder=5, label="MLE")

        if credal_flag:
            coord_max = np.max(coords[:, 0])
            coord_min = np.min(coords[:, 0])
            ax.fill_betweenx(
                y_marg,
                coord_max,
                coord_min,
                color=cfg.BLUE,
                alpha=cfg.FILL_ALPHA,
                zorder=2,
                label="Credal-band",
            )

        ax.scatter(coords[:, 0], coords[:, 1], color=cfg.BLUE, zorder=3, label="Probabilities")

        ax.axis("off")
        ax.set_ylim((-0.2, 0.2))

        y_anchor = -0.07
        x_beg = 0
        x_mid = 0.5  # noqa: F841
        x_end = 1

        ax.text(x_beg, y_anchor - 0.07, f"{labels[0]}", ha="center", va="top")
        ax.text(x_end, y_anchor - 0.07, f"{labels[1]}", ha="center", va="top")

        tick_values = np.linspace(0.0, 1.0, 11)
        tick_length = 0.02
        label_offset = -0.05
        e1 = np.array([0.0, 0.0])
        e2 = np.array([1.0, 0.0])
        edges = [(e1, e2, "A")]

        def lerp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
            """Linear Interpolation for line values.

            Args:
                p: NumPy Array representing starting point.
                q: NumPy Array representing ending point.
                t: Interpolation parameter in [0, 1], where 0 returns p and 1 returns q.

            Returns:
                Interpolated point between p and q.
            """
            return (1 - t) * p + t * q

        for p, q, axis_name in edges:  # noqa: B007
            edge_vec = q - p
            normal = np.array([-edge_vec[1], edge_vec[0]])
            normal = normal / np.linalg.norm(normal)

            for t in tick_values[1:-1]:
                pos = lerp(p, q, t)

                tick_start = pos - normal * (tick_length / 2)
                tick_end = pos + normal * (tick_length / 2)
                ax.plot(
                    [tick_start[0], tick_end[0]],
                    [tick_start[1], tick_end[1]],
                    color=cfg.BLACK,
                )
                label_pos = pos + normal * label_offset
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    f"{t:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        ax.set_title(title, pad=20)
        ax.legend(loc="upper left")

        return ax
