"""Plotting for 3 class probabilities."""

from __future__ import annotations

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

import probly.visualization.config as cfg


class TernaryVisualizer:
    """Class to collect all the geometric plots."""

    def __init__(self) -> None:
        """Initialize the class."""

    def probs_to_coords_3d(self, probs: np.ndarray) -> tuple:
        """Convert ternary probabilities to 2D coordinates.

        Args:
        probs: the ternary probabilities.

        return: Tuple containing the 2D coordinates.
        """
        _, p2, p3 = probs
        x = p2 + 0.5 * p3
        y = (np.sqrt(3) / 2) * p3
        return x, y

    def label_corners_and_vertices(
        self,
        ax: plt.Axes,
        v1: np.array,
        v2: np.array,
        v3: np.array,
        labels: list[str],
    ) -> None:
        """Labeling the corners and vertices."""
        c1 = f"{labels[0]}"
        c2 = f"{labels[1]}"
        c3 = f"{labels[2]}"
        offset_x = 0.06
        ax.text(v1[0] + 0.02, v1[1] - offset_x, c1, ha="right", va="top", fontsize=12)
        ax.text(v2[0] + 0.02, v2[1] - offset_x, c2, ha="left", va="top", fontsize=12)
        ax.text(v3[0], v3[1] + offset_x, c3, ha="center", va="bottom", fontsize=12)

        edge_lable = "0.0 / 1.0"
        ax.text(v1[0], v1[1], edge_lable, ha="right", va="top", fontsize=8)
        ax.text(v2[0], v2[1], edge_lable, ha="left", va="top", fontsize=8)
        ax.text(v3[0], v3[1], edge_lable, ha="center", va="bottom", fontsize=8)

    def ternary_plot(
        self,
        probs: np.ndarray,
        labels: list[str],
        title: str = "Ternary Plot (3 Classes)",
        ax: plt.Axes = None,
        plot_hull: bool = True,
        plot_minmax: bool = True,
        **scatter_kwargs: object,
    ) -> plt.Axes:
        """Plot ternary scatter points.

        Args:
        probs: the ternary probabilities.
        labels: the labels of the ternary points.
        title: fixed title of the plot.
        scatter_kwargs: keyword arguments passed to scatter_kwargs.
        ax: matplotlib axes.Axes to plot on.
        plot_hull: bool defaulted to true, which optionally draws a convex hull.
        plot_minmax: bool defaulted to true, which optionally draws upper and lower probability envelopes.

        returns: Ternary plot with scattered points.
        """
        coords = np.array([self.probs_to_coords_3d(x) for x in probs])

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        def lerp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
            """Linear Interpolation for line values."""
            return (1 - t) * p + t * q

        # Draw triangle
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, np.sqrt(3) / 2])

        edges = [(v1, v2, "axis A"), (v2, v3, "axis B"), (v3, v1, "axis C")]

        triangle_x = [v1[0], v2[0], v3[0], v1[0]]
        triangle_y = [v1[1], v2[1], v3[1], v1[1]]

        ax.plot(triangle_x, triangle_y, color=cfg.BLACK)
        ax.axis("off")

        self.label_corners_and_vertices(ax, v1, v2, v3, labels)

        verts = np.array([v1, v2, v3])  # noqa: F841
        # tick_values are set in a way that they won't interfere at the edges
        tick_values = np.linspace(0.0, 1.0, 11)
        tick_length = 0.01
        label_offset = -0.05

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
        ax.set_aspect("equal", "box")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3) / 2)

        # Scatter points
        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

        ax.set_title(title, pad=20, y=-0.2)

        if plot_hull:
            self.plot_convex_hull(probs, ax=ax)

        # Optionally draw second order max/min envelope lines
        if plot_minmax:
            self.plot_minmax_lines(probs, ax=ax)

        return ax

    def draw_prob_line(self, ax: plt.Axes) -> None:
        """Mini demo: draw probability axes through one fixed example point.

        Args:
        ax: Axes to draw on
        Draws the three isolines through (a,b,c):
        - constant a (parallel to BC)
        - constant b (parallel to AC)
        - constant c (parallel to AB).
        """
        # hard coded demo
        example = np.array([0.25, 0.55, 0.20])
        x, y = self.probs_to_coords_3d(example)
        ax.scatter([x], [y], color=cfg.RED, s=30, zorder=5)

        a, b, c = example

        # helper to draw lines
        def seg(p: np.ndarray, q: np.ndarray, *, color: str, lw: float = 2.0, alpha: float = 1.0) -> None:
            x1, y1 = self.probs_to_coords_3d(p)
            x2, y2 = self.probs_to_coords_3d(q)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=4)

        # helper to draw text with border
        def mid_lable(p: np.ndarray, q: np.ndarray, text: str, *, fontsize: int = 9) -> None:
            x1, y1 = self.probs_to_coords_3d(p)
            x2, y2 = self.probs_to_coords_3d(q)
            xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            txt = ax.text(xm, ym, text, ha="center", va="center", color=cfg.WHITE, fontsize=fontsize, zorder=6)
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground=cfg.BLACK)])

        p_ac = np.array([a, 0.0, 1.0 - a])
        seg(example, p_ac, color=cfg.BLUE, lw=1, alpha=1.0)
        mid_lable(example, p_ac, f"a={a:.2f}")
        p_ba = np.array([1.0 - b, b, 0.0])
        seg(p_ba, example, color=cfg.BLUE, lw=1, alpha=1.0)
        mid_lable(p_ba, example, f"b={b:.2f}")
        p_cb = np.array([0.0, 1.0 - c, c])
        seg(example, p_cb, color=cfg.BLUE, lw=1, alpha=1.0)
        mid_lable(example, p_cb, f"c={c:.2f}")

    def plot_convex_hull(
        self,
        probs: np.ndarray,
        ax: plt.Axes = None,
    ) -> plt.Axes:
        """Draw the convex hull around the points.

        Handles special cases:
        - 1 point (degenerate)
        - 2 points (line segment)
        - >= 3 points (polygon)

        Args:
        probs: Array of probabilities
        ax: Axes to draw on
        facecolor: Color of the convex hull
        alpha: Opacity of the convex hull
        edgecolor: Color of the outline
        linewidth: Width of the convex hull

        returns: Plot with convex hull.
        """
        coords = np.array([self.probs_to_coords_3d(p) for p in probs])

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        # Handle degenerate cases
        unique = np.unique(coords, axis=0)

        if len(unique) == 1:
            # Single point — no hull possible
            ax.scatter(unique[:, 0], unique[:, 1], color=cfg.RED, s=80)
            return ax

        if len(unique) == 2:
            # Two distinct points — hull is a line segment
            ax.plot(unique[:, 0], unique[:, 1], color=cfg.HULL_EDGE, linewidth=cfg.HULL_LINE_WIDTH)
            return ax

        # Try to compute convex hull
        try:
            hull = ConvexHull(coords)

            hull_pts = coords[hull.vertices]

            poly = plt.Polygon(
                hull_pts,
                closed=True,
                facecolor=cfg.HULL_FACE,
                edgecolor=cfg.HULL_EDGE,
                alpha=cfg.FILL_ALPHA,
                linewidth=cfg.HULL_LINE_WIDTH,
            )
            ax.add_patch(poly)

        except Exception:  # noqa: BLE001
            # Remaining degeneracy: 3+ collinear points
            # Get endpoints via projection on PCA / longest distance
            dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=cfg.HULL_EDGE,
                linewidth=cfg.HULL_LINE_WIDTH,
            )
        self.draw_prob_line(ax)
        return ax

    def _draw_constant_probability_line(
        self,
        ax: plt.Axes,
        index: int,
        value: float,
        color: str = "red",
        linestyle: str = "--",
    ) -> None:
        """Draw a line of constant probability p[index] = value.

        The line is parallel to the edge opposite the corresponding vertex.
        """
        if value <= 0 or value >= 1:
            return  # Degenerate, coincides with triangle boundary

        # Remaining mass for the other two components
        remaining = 1.0 - value

        # Two extreme endpoints
        p_start = np.zeros(3)
        p_end = np.zeros(3)

        p_start[index] = value
        p_end[index] = value

        other = [i for i in range(3) if i != index]

        p_start[other[0]] = remaining
        p_start[other[1]] = 0.0

        p_end[other[0]] = 0.0
        p_end[other[1]] = remaining

        x1, y1 = self.probs_to_coords_3d(p_start)
        x2, y2 = self.probs_to_coords_3d(p_end)

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linestyle=linestyle,
            linewidth=cfg.MIN_MAX_LINE_WIDTH,
            alpha=cfg.MIN_MAX_ALPHA,
        )

    def plot_minmax_lines(
        self,
        probs: np.ndarray,
        ax: plt.Axes,
    ) -> None:
        """Draw min/max probability lines for each class.

        Up to 6 lines total (min & max for 3 classes).
        """
        p_min = probs.min(axis=0)
        p_max = probs.max(axis=0)

        for i in range(3):
            self._draw_constant_probability_line(
                ax=ax,
                index=i,
                value=p_min[i],
                color=cfg.RED,
                linestyle=cfg.MIN_MAX_LINESTYLE_1,
            )
            self._draw_constant_probability_line(
                ax=ax,
                index=i,
                value=p_max[i],
                color=cfg.BLUE,
                linestyle=cfg.MIN_MAX_LINESTYLE_2,
            )
