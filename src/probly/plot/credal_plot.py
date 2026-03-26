"""Credal set ternary plot."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import mpltern  # noqa: F401
import numpy as np

from .config import PlotConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpltern import TernaryAxes

    from probly.representation.credal_set.array import ArrayProbabilityIntervalsCredalSet

__all__ = ["plot_credal_set"]

_NUM_TERNARY_CLASSES = 3


def _compute_interval_vertices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Compute vertices of the feasible region on the 3-simplex from probability intervals.

    Enumerates all combinations of lower/upper bounds for pairs of dimensions,
    derives the third via the simplex constraint (sum = 1), and keeps only
    vertices that satisfy all bounds. Returns vertices sorted by polar angle
    for correct polygon winding.

    Args:
        lower: Lower probability bounds, shape (3,).
        upper: Upper probability bounds, shape (3,).

    Returns:
        Array of shape (n_vertices, 3) with feasible vertices sorted for polygon rendering.

    Raises:
        ValueError: If no feasible vertices are found.
    """
    vertices: list[list[float]] = []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
        for x in [lower[i], upper[i]]:
            for y in [lower[j], upper[j]]:
                z = 1.0 - x - y
                if lower[k] - 1e-9 <= z <= upper[k] + 1e-9:
                    p = [0.0, 0.0, 0.0]
                    p[i] = x
                    p[j] = y
                    p[k] = z
                    vertices.append(p)

    if not vertices:
        msg = "No feasible vertices found. Check that the probability intervals are valid and overlap the simplex."
        raise ValueError(msg)

    pts = np.array(vertices)
    # Remove near-duplicate vertices
    pts = np.unique(np.round(pts, decimals=10), axis=0)

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def plot_credal_set(
    data: ArrayProbabilityIntervalsCredalSet,
    *,
    title: str | None = None,
    labels: list[str] | None = None,
    config: PlotConfig | None = None,
    show: bool = False,
) -> Axes:
    """Plot a ProbabilityInterval credal set as a polytope on a ternary simplex.

    Only supports 3-class credal sets. For batched inputs, each credal set
    in the batch is overlaid on the same axes with a distinct color.

    Args:
        data: The credal set to plot. Must have ``num_classes == 3``.
        title: Title of the plot.
        labels: Corner labels for the three classes. Defaults to ``["C1", "C2", "C3"]``.
        config: Plot configuration. Defaults to ``PlotConfig()``.
        show: Whether to call ``plt.show()`` after plotting.

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If ``data.num_classes != 3`` or labels length doesn't match.
    """
    if data.num_classes != _NUM_TERNARY_CLASSES:
        msg = f"plot_credal_set only supports 3-class credal sets, got {data.num_classes} classes."
        raise ValueError(msg)

    config = config or PlotConfig()
    labels = labels or ["C1", "C2", "C3"]

    if len(labels) != _NUM_TERNARY_CLASSES:
        msg = f"Expected 3 labels, got {len(labels)}."
        raise ValueError(msg)

    fig = plt.figure(figsize=config.figure_size, dpi=config.dpi)
    ternary_ax = cast("TernaryAxes", fig.add_subplot(projection="ternary"))

    # Flatten batch dimensions for iteration
    lower_all = data.lower_bounds.reshape(-1, _NUM_TERNARY_CLASSES)
    upper_all = data.upper_bounds.reshape(-1, _NUM_TERNARY_CLASSES)
    n_sets = lower_all.shape[0]

    for idx in range(n_sets):
        vertices = _compute_interval_vertices(lower_all[idx], upper_all[idx])
        color = config.color(idx)

        # Close the polygon
        closed = np.vstack([vertices, vertices[0]])
        ternary_ax.fill(closed[:, 0], closed[:, 1], closed[:, 2], alpha=config.fill_alpha, color=color)
        ternary_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, linewidth=config.line_width)

    ternary_ax.set_tlabel(labels[0], fontsize=config.label_fontsize)
    ternary_ax.set_llabel(labels[1], fontsize=config.label_fontsize)
    ternary_ax.set_rlabel(labels[2], fontsize=config.label_fontsize)

    if title is not None:
        ternary_ax.set_title(title, fontsize=config.title_fontsize)

    if show:
        plt.show()

    return ternary_ax
