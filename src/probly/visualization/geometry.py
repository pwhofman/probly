"""So far function to show a terniary plot. With example."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class CredalVisualizer:
    """Class to collect all the geometric plots."""

    def __init__(self) -> None:
        """Function to pass object."""

    def probs_to_coords(self, probs: np.ndarray) -> tuple:
        """Function to convert ternary probabilities to coordinates.

        Args:
            self
            probs: Numpy array of ternary probabilities.

        returns a tuple of the coordinates.
        """
        p1, p2, p3 = probs
        x = p2 + 0.5 * p3
        y = (np.sqrt(3) / 2) * p3
        return x, y

    def ternary_plot(
        self,
        probs: np.ndarray,
        ax: mpl.axes.Axes = None,
        **scatter_kwargs: mpl.Kwargs,
    ) -> mpl.axes.Axes:
        """Function to plot ternary plots.

        Args:
            self:
            probs: Numpy array of ternary probabilities
            ax: Axes to plot on.
            **scatter_kwargs: Keyword arguments to pass to scatter function.

        returns an axes object.
        """
        msg = "Inpu must have 3 dimensions."
        if probs.shape[1] != 3:
            raise ValueError(msg)

        coords = np.array([self.probs_to_coords(p) for p in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        verts = np.array(
            [
                [0, 0],
                [1, 0],
                [0.5, np.sqrt(3) / 2],
            ],
        )

        triangle = plt.Polygon(verts, closed=True, fill=False)
        ax.add_patch(triangle)

        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

        return ax


points = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.4, 0.3, 0.3],
        [0.1, 0.1, 0.8],
    ],
)

viz = CredalVisualizer()
viz.ternary_plot(points, color="blue", s=50)
plt.show()
