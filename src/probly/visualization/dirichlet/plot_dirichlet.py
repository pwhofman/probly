"""Plotting Dirichlet distributions on a ternary simplex."""

from __future__ import annotations

from math import gamma
from typing import TYPE_CHECKING, Any

from matplotlib import tri
import matplotlib.pyplot as plt
import numpy as np

import probly.visualization.config as cfg

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class DirichletTernaryVisualizer:
    """Class to collect ternary Dirichlet plots."""

    def __init__(self) -> None:
        """Initialize the class."""

    def triangle_corners(self) -> np.ndarray:
        """Return the corners of the equilateral ternary triangle."""
        return np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2],
            ]
        )

    def xy_to_barycentric(self, xy: np.ndarray, tol: float = 1e-4) -> np.ndarray:
        """Convert Cartesian coordinates to barycentric coordinates.

        Args:
            xy: Cartesian coordinates inside the triangle.
            tol: Numerical tolerance to avoid simplex boundaries.

        Returns:
            Barycentric coordinates.
        """
        corners = self.triangle_corners()

        def to3(v: np.ndarray) -> np.ndarray:
            """Promote 2D vector to 3D.

            Args:
                v: A 2D vector of shape (2,).

            Returns:
                A 3D vector of shape (3,) with z-component set to 0.0.
            """
            return np.array([v[0], v[1], 0.0])

        area = float(
            0.5
            * np.linalg.norm(
                np.cross(
                    to3(corners[1] - corners[0]),
                    to3(corners[2] - corners[0]),
                )
            )
        )

        pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]

        def tri_area(point: np.ndarray, pair: np.ndarray) -> float:
            """Compute area of a triangle defined by 'point' and two vertices.

            Args:
                point: A 2D point (shape (2,)).
                pair: Two 2D vertices (shape (2, 2)).

            Returns:
                The triangle area as a float.
            """
            area = 0.5 * np.linalg.norm(
                np.cross(
                    to3(pair[0] - point),
                    to3(pair[1] - point),
                )
            )
            return float(area)

        coords = np.array([tri_area(xy, p) for p in pairs]) / area
        clipped_coords = np.clip(coords, tol, 1.0 - tol)
        return clipped_coords

    class Dirichlet:
        """Dirichlet distribution."""

        def __init__(self, alpha: np.ndarray) -> None:
            """Initialize the distribution.

            Args:
            alpha: Dirichlet concentration parameters.
            """
            self.alpha = np.asarray(alpha)
            self.coef = gamma(np.sum(self.alpha)) / np.prod([gamma(a) for a in self.alpha])

        def pdf(self, x: np.ndarray) -> float:
            """Compute the Dirichlet pdf.

            Args:
                x: Barycentric coordinates.

            Returns:
                The PDF value at x as a float.
            """
            return float(self.coef * np.prod([xx ** (aa - 1) for xx, aa in zip(x, self.alpha, strict=False)]))

    def label_corners_and_vertices(
        self,
        ax: Axes,
        labels: list[str],
    ) -> None:
        """Label corners, vertices, and edge ticks.

        Args:
            ax: Matplotlib Axes to annotate.
            labels: Three labels corresponding to the simplex corners.
        """
        v1, v2, v3 = self.triangle_corners()

        # Corner labels
        offset = 0.06
        ax.text(v1[0] + 0.02, v1[1] - offset, labels[0], ha="right", va="top", fontsize=12)
        ax.text(v2[0] + 0.02, v2[1] - offset, labels[1], ha="left", va="top", fontsize=12)
        ax.text(v3[0], v3[1] + offset, labels[2], ha="center", va="bottom", fontsize=12)

        # Vertex labels
        edge_label = "0.0 / 1.0"
        ax.text(v1[0], v1[1], edge_label, ha="right", va="top", fontsize=8)
        ax.text(v2[0], v2[1], edge_label, ha="left", va="top", fontsize=8)
        ax.text(v3[0], v3[1], edge_label, ha="center", va="bottom", fontsize=8)

        # Edge ticks
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        tick_values = np.linspace(0.0, 1.0, 11)
        tick_length = 0.01
        label_offset = -0.05

        def lerp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
            """Linearly interpolate between two points.

            Args:
                p: Start point.
                q: End point.
                t: Interpolation parameter.

            Returns:
                Interpolated point.
            """
            return (1.0 - t) * p + t * q

        for p, q in edges:
            edge_vec = q - p
            normal = np.array([-edge_vec[1], edge_vec[0]])
            normal /= np.linalg.norm(normal)

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

    def dirichlet_plot(  # noqa: D417
        self,
        alpha: np.ndarray,
        labels: list[str],
        title: str,
        ax: Axes | None = None,
        subdiv: int = 7,
        nlevels: int = 200,
        **contour_kwargs: Any,  # noqa: ANN401
    ) -> Axes:
        """Plot Dirichlet pdf contours on a ternary simplex.

        Args:
            alpha: Dirichlet concentration parameters.
            labels: Labels of the ternary corners.
            title: Title of the plot.
            ax: Matplotlib axes.Axes to plot on.
            subdiv: Triangular mesh subdivision depth.
            nlevels: Number of contour levels.
            cmap: Matplotlib colormap.

        Returns:
            Ternary plot with Dirichlet contours.
        """
        corners = self.triangle_corners()
        triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

        refiner = tri.UniformTriRefiner(triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)

        dist = self.Dirichlet(alpha)

        pvals = [dist.pdf(self.xy_to_barycentric(np.array(xy))) for xy in zip(trimesh.x, trimesh.y, strict=False)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.subplots_adjust(bottom=0.25)

        ax.tricontourf(
            trimesh,
            pvals,
            nlevels,
            cmap=cfg.PROBLY_CMAP,
            **contour_kwargs,
        )

        ax.plot(
            [corners[0, 0], corners[1, 0], corners[2, 0], corners[0, 0]],
            [corners[0, 1], corners[1, 1], corners[2, 1], corners[0, 1]],
            color=cfg.BLACK,
        )

        self.label_corners_and_vertices(ax, labels)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3) / 2)
        ax.axis("off")
        ax.set_title(title, pad=40)

        return ax
