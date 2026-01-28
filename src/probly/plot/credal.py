"""Collection of credal plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mpltern  # noqa: F401, required for ternary projection, do not remove
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def simplex_plot(probs: np.ndarray) -> tuple[Figure, Axes]:
    """Plot probability distributions on the simplex.

    Args:
        probs: numpy.ndarray of shape (n_instances, n_classes)

    Returns:
        fig: matplotlib figure
        ax: matplotlib axes
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="ternary")
    ax.scatter(probs[:, 0], probs[:, 1], probs[:, 2])
    return fig, ax


def credal_set_plot(probs: np.ndarray) -> tuple[Figure, Axes]:
    """Plot credal sets based on intervals of lower and upper probabilities.

    Args:
        probs: numpy.ndarray of shape (n_samples, n_classes)

    Returns:
        fig: matplotlib figure
        ax: matplotlib axes
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="ternary")

    lower_probs = np.min(probs, axis=0)
    upper_probs = np.max(probs, axis=0)
    lower_idxs = np.argmin(probs, axis=0)
    upper_idxs = np.argmax(probs, axis=0)
    edge_probs = np.vstack((probs[lower_idxs], probs[upper_idxs]))

    vertices_ = []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
        for x in [lower_probs[i], upper_probs[i]]:
            for y in [lower_probs[j], upper_probs[j]]:
                z = 1 - x - y
                if lower_probs[k] <= z <= upper_probs[k]:
                    prob = [0, 0, 0]
                    prob[i] = x
                    prob[j] = y
                    prob[k] = z
                    vertices_.append(prob)
    vertices = np.array(vertices_)

    if len(vertices) > 0:
        center = np.mean(vertices, axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        vertices = vertices[np.argsort(angles)]
        ax.scatter(probs[:, 0], probs[:, 1], probs[:, 2])
        vertices_closed = np.vstack([vertices, vertices[0]])
        ax.fill(vertices_closed[:, 0], vertices_closed[:, 1], vertices_closed[:, 2])
        ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], vertices_closed[:, 2])
        ax.scatter(edge_probs[:, 0], edge_probs[:, 1], edge_probs[:, 2])
    else:
        msg = "The set of vertices is empty. Please check the probabilities in the credal set."
        raise ValueError(msg)

    return fig, ax
