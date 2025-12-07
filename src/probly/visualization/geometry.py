from __future__ import annotations  # noqa: D100

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


class CredalVisualizer:
    """Class to collect all the geometric plots."""

    def __init__(self) -> None:  # noqa: D107
        pass

    def probs_to_coords(self, probs: np.ndarray) -> tuple:
        """Convert ternary probabilities to 2D coordinates."""
        p1, p2, p3 = probs  # noqa: RUF059
        x = p2 + 0.5 * p3
        y = (np.sqrt(3) / 2) * p3
        return x, y

    def ternary_plot(
        self,
        probs: np.ndarray,
        ax: mpl.axes.Axes = None,
        **scatter_kwargs: mpl.Kwargs,
    ) -> mpl.axes.Axes:
        """Plot ternary scatter points."""
        msg = "Input must have 3 dimensions."
        if probs.shape[1] != 3:
            raise ValueError(msg)

        coords = np.array([self.probs_to_coords(p) for p in probs])
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # noqa: RUF059

        # Draw triangle
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        v3 = np.array([0.5, np.sqrt(3) / 2])
        c1 = "Class 1"
        c2 = "Class 2"
        c3 = "Class 3"
        ax.text(v1[0], v1[1], c1, ha="right", va="top", fontsize=12)
        ax.text(v2[0], v2[1], c2, ha="left", va="top", fontsize=12)
        ax.text(v3[0], v3[1], c3, ha="center", va="bottom", fontsize=12)

        ax.axis("off")
        verts = np.array([v1, v2, v3])

        triangle = plt.Polygon(verts, closed=True, fill=False)
        ax.add_patch(triangle)

        # Scatter points
        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

        return ax

    def plot_convex_hull(
        self,
        probs: np.ndarray,
        ax: mpl.axes.Axes = None,
        facecolor: str = "lightgreen",
        alpha: float = 0.4,
        edgecolor: str = "green",
        linewidth: float = 2.0,
    ) -> mpl.axes.Axes:
        """Draw the convex hull around ternary points.
        Handles special cases:
        - 1 point (degenerate)
        - 2 points (line segment)
        - ≥3 points (polygon).
        """  # noqa: D205
        coords = np.array([self.probs_to_coords(p) for p in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # noqa: RUF059

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

viz = CredalVisualizer()

ax = viz.ternary_plot(points, color="blue", s=50)
viz.plot_convex_hull(points, ax=ax)

plt.show()
