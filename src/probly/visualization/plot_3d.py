"""Class to collect the different types of plots."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


class TernaryVisualizer:
    """Class to collect all the geometric plots."""

    def __init__(self) -> None:
        """Initialize the class."""

    def probs_to_coords(self, probs: np.ndarray) -> tuple:
        """Convert ternary probabilities to 2D coordinates.

        Args:
        probs: the ternary probabilities.

        return: Tuple containing the 2D coordinates.
        """
        p1, p2, p3 = probs
        x = p2 + 0.5 * p3
        y = (np.sqrt(3) / 2) * p3
        return x, y

    def ternary_plot(
        self,
        probs: np.ndarray,
        labels: list[str] | None = None,
        title: str = "Ternary Plot (3 Classes)",
        ax: plt.Axes = None,
        **scatter_kwargs: mpl.Kwargs,
    ) -> plt.Axes:
        """Plot ternary scatter points.

        Args:
        probs: the ternary probabilities.
        labels: the labels of the ternary points.
        scatter_kwargs: keyword arguments passed to scatter_kwargs.
        ax: matplotlib axes.Axes to plot on.

        returns: Ternary plot with scattered points.
        """
        if probs.shape[1] != 3:
            raise ValueError

        coords = np.array([self.probs_to_coords(x) for x in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

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

        ax.plot(triangle_x, triangle_y, color="black")
        ax.axis("off")

        n_classes = probs.shape[-1]

        if labels is None:
            labels = [f"C{i + 1}" for i in range(n_classes)]

        if len(labels) != n_classes:
            msg = f"Number of labels ({len(labels)}) must match number of classes ({n_classes})."
            raise ValueError(msg)

        c1 = f"{labels[0]}"
        c2 = f"{labels[1]}"
        c3 = f"{labels[2]}"
        offset_x = 0.06
        ax.text(v1[0] + 0.02, v1[1] - offset_x, c1, ha="right", va="top", fontsize=12)
        ax.text(v2[0] + 0.02, v2[1] - offset_x, c2, ha="left", va="top", fontsize=12)
        ax.text(v3[0], v3[1] + offset_x, c3, ha="center", va="bottom", fontsize=12)

        verts = np.array([v1, v2, v3])  # noqa: F841
        # tick_values are set in a way that they won't interfere at the edges
        tick_values = np.linspace(0.1, 0.90, 11)
        tick_length = 0.01
        label_offset = -0.05
        edge_lable = "0.0 / 1.0"
        ax.text(v1[0], v1[1], edge_lable, ha="right", va="top", fontsize=8)
        ax.text(v2[0], v2[1], edge_lable, ha="left", va="top", fontsize=8)
        ax.text(v3[0], v3[1], edge_lable, ha="center", va="bottom", fontsize=8)
        for p, q, axis_name in edges:  # noqa: B007
            edge_vec = q - p
            normal = np.array([-edge_vec[1], edge_vec[0]])
            normal = normal / np.linalg.norm(normal)

            for t in tick_values:
                pos = lerp(p, q, t)

                tick_start = pos - normal * (tick_length / 2)
                tick_end = pos + normal * (tick_length / 2)
                ax.plot(
                    [tick_start[0], tick_end[0]],
                    [tick_start[1], tick_end[1]],
                    color="black",
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


        ax.set_title(title, pad = 20, y = -0.2)

        return ax

    def interval_plot(self) -> None:
        """To be implemented. Plot for 2 classes."""
        return

    def spider_plot(self) -> None:
        """To be implemented. Plot for more than 3 classes."""
        return

    def plot_convex_hull(
        self,
        probs: np.ndarray,
        ax: mpl.axes.Axes = None,
        facecolor: str = "lightgreen",
        alpha: float = 0.4,
        edgecolor: str = "green",
        linewidth: float = 2.0,
    ) -> mpl.axes.Axes:
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
        coords = np.array([self.probs_to_coords(p) for p in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Handle degenerate cases
        unique = np.unique(coords, axis=0)

        if len(unique) == 1:
            # Single point — no hull possible
            ax.scatter(unique[:, 0], unique[:, 1], color="green", s=80)
            return ax

        if len(unique) == 2:
            # Two distinct points — hull is a line segment
            ax.plot(unique[:, 0], unique[:, 1], color=edgecolor, linewidth=linewidth)
            return ax

        # Try to compute convex hull
        try:
            hull = ConvexHull(coords)

            hull_pts = coords[hull.vertices]

            poly = plt.Polygon(
                hull_pts,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
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
                color=edgecolor,
                linewidth=linewidth,
            )

        return ax


points = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.4, 0.3, 0.3],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.3, 0.1, 0.6],
        [0.33, 0.33, 0.34],
    ],
)

ter = TernaryVisualizer()

ax = ter.ternary_plot(points)
ter.plot_convex_hull(points, ax=ax)

plt.show()
