"""Credal set ternary plot."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import mpltern  # noqa: F401
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from lazy_dispatch import lazydispatch
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

from .config import PlotConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpltern import TernaryAxes

    type ArrayCredalSet = (
        ArrayProbabilityIntervalsCredalSet
        | ArrayDistanceBasedCredalSet
        | ArrayConvexCredalSet
        | ArrayDiscreteCredalSet
        | ArraySingletonCredalSet
    )

__all__ = ["plot_credal_set"]

_NUM_TERNARY_CLASSES = 3


# ── Type-dispatched helpers ──────────────────────────────────────────────────


@lazydispatch
def _get_num_classes(data: ArrayCredalSet) -> int:
    """Extract the number of classes from any Array credal set type.

    Args:
        data: Any Array credal set instance.

    Returns:
        The number of classes.

    Raises:
        NotImplementedError: If the credal set type is not supported.
    """
    msg = f"Unsupported credal set type: {type(data).__name__}"
    raise NotImplementedError(msg)


@_get_num_classes.register(ArrayProbabilityIntervalsCredalSet)
def _(data: ArrayProbabilityIntervalsCredalSet) -> int:
    return data.num_classes


@_get_num_classes.register(ArrayDistanceBasedCredalSet)
def _(data: ArrayDistanceBasedCredalSet) -> int:
    return data.nominal.shape[-1]


@_get_num_classes.register(ArrayConvexCredalSet)
@_get_num_classes.register(ArrayDiscreteCredalSet)
@_get_num_classes.register(ArraySingletonCredalSet)
def _(data: ArrayConvexCredalSet | ArrayDiscreteCredalSet | ArraySingletonCredalSet) -> int:
    return data.array.shape[-1]


# ── Geometry helpers ─────────────────────────────────────────────────────────


def _sort_vertices_by_angle(pts: np.ndarray) -> np.ndarray:
    """Sort a set of 3-simplex points by polar angle around their centroid.

    Args:
        pts: Array of shape (n, 3) with points on the simplex.

    Returns:
        The same points reordered by ascending polar angle for polygon winding.
    """
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def _compute_convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Compute the ordered convex hull boundary from a set of 3-simplex vertices.

    For fewer than 3 points the hull degenerates, so the points are returned
    sorted by angle instead. For 3 or more points scipy ``ConvexHull`` is used
    on the first two coordinates (the third is redundant on the simplex), with
    a fallback to angle sorting if the hull computation fails.

    Args:
        points: Array of shape (n, 3) with points on the 3-simplex.

    Returns:
        Array of shape (m, 3) with hull vertices in polygon winding order.
    """
    if len(points) < 3:
        return _sort_vertices_by_angle(points)

    pts_2d = points[:, :2]
    try:
        hull = ConvexHull(pts_2d)
        return points[hull.vertices]
    except QhullError:
        return _sort_vertices_by_angle(points)


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

    return _sort_vertices_by_angle(pts)


def _draw_polygon(
    ternary_ax: TernaryAxes,
    vertices: np.ndarray,
    color: object,
    config: PlotConfig,
) -> None:
    """Render a filled polygon and its outline on ternary axes.

    Args:
        ternary_ax: The mpltern ternary axes to draw on.
        vertices: Array of shape (n, 3) with polygon vertices in winding order.
        color: Matplotlib color for both fill and outline.
        config: Plot configuration supplying fill_alpha and line_width.
    """
    closed = np.vstack([vertices, vertices[0]])
    ternary_ax.fill(closed[:, 0], closed[:, 1], closed[:, 2], alpha=config.fill_alpha, color=color)
    ternary_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, linewidth=config.line_width)


@lazydispatch
def _draw_credal_set(
    data: ArrayCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
) -> None:
    """Draw a credal set on ternary axes.

    Args:
        data: The credal set to draw.
        ternary_ax: The mpltern ternary axes to draw on.
        config: Plot configuration.

    Raises:
        NotImplementedError: If no handler is registered for the given type.
    """
    msg = f"Unsupported credal set type: {type(data).__name__}"
    raise NotImplementedError(msg)


@_draw_credal_set.register(ArraySingletonCredalSet)
def _draw_singleton(
    data: ArraySingletonCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
) -> None:
    """Draw an ArraySingletonCredalSet as scatter markers on the ternary axes.

    Each batch element is drawn as a single point.

    Args:
        data: The singleton credal set.
        ternary_ax: The mpltern ternary axes to draw on.
        config: Plot configuration.
    """
    arr = data.array
    pts = arr.reshape(-1, _NUM_TERNARY_CLASSES)
    n_sets = pts.shape[0]

    for idx in range(n_sets):
        color = config.color(idx)
        p = pts[idx]
        ternary_ax.scatter(p[0:1], p[1:2], p[2:3], color=color, s=config.marker_size, zorder=3)


@_draw_credal_set.register(ArrayProbabilityIntervalsCredalSet)
def _draw_intervals(
    data: ArrayProbabilityIntervalsCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
) -> None:
    """Draw an ArrayProbabilityIntervalsCredalSet as filled polygons on the ternary axes.

    Each element in the batch is drawn as a separate filled polygon.

    Args:
        data: The probability intervals credal set.
        ternary_ax: The mpltern ternary axes to draw on.
        config: Plot configuration.
    """
    lower_all = data.lower_bounds.reshape(-1, _NUM_TERNARY_CLASSES)
    upper_all = data.upper_bounds.reshape(-1, _NUM_TERNARY_CLASSES)
    n_sets = lower_all.shape[0]

    for idx in range(n_sets):
        vertices = _compute_interval_vertices(lower_all[idx], upper_all[idx])
        color = config.color(idx)
        _draw_polygon(ternary_ax, vertices, color, config)


@_draw_credal_set.register(ArrayDistanceBasedCredalSet)
def _draw_distance_based(
    data: ArrayDistanceBasedCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
) -> None:
    """Draw an ArrayDistanceBasedCredalSet as filled polygons with nominal point markers.

    Lower and upper bounds are derived from the nominal distribution and radius,
    then treated identically to probability intervals. The nominal point itself
    is also drawn as a scatter marker.

    Args:
        data: The distance-based credal set.
        ternary_ax: The mpltern ternary axes to draw on.
        config: Plot configuration.
    """
    lower_all = data.lower().reshape(-1, _NUM_TERNARY_CLASSES)
    upper_all = data.upper().reshape(-1, _NUM_TERNARY_CLASSES)
    nominal_all = data.nominal.reshape(-1, _NUM_TERNARY_CLASSES)
    n_sets = lower_all.shape[0]

    for idx in range(n_sets):
        color = config.color(idx)
        vertices = _compute_interval_vertices(lower_all[idx], upper_all[idx])
        _draw_polygon(ternary_ax, vertices, color, config)

        nom = nominal_all[idx]
        ternary_ax.scatter(nom[0:1], nom[1:2], nom[2:3], color=color, s=config.marker_size, zorder=3)


@_draw_credal_set.register(ArrayDiscreteCredalSet | ArrayConvexCredalSet)
def _draw_vertex_set(
    data: ArrayConvexCredalSet | ArrayDiscreteCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
) -> None:
    """Draw an ArrayConvexCredalSet or ArrayDiscreteCredalSet as a convex hull polygon.

    For each batch element the convex hull of the vertices/members is drawn as a
    filled polygon with scatter markers at each vertex. If fewer than 3 unique
    points are present the polygon is omitted (a line is drawn for exactly 2
    points, markers only for 1).

    Args:
        data: The convex or discrete credal set.
        ternary_ax: The mpltern ternary axes to draw on.
        config: Plot configuration.
    """
    # Shape: (..., num_vertices, num_classes) → flatten batch dims
    arr = data.array.reshape(-1, data.array.shape[-2], _NUM_TERNARY_CLASSES)
    n_sets = arr.shape[0]

    for idx in range(n_sets):
        color = config.color(idx)
        pts = arr[idx]  # (num_vertices, 3)

        # Remove near-duplicates
        pts = np.unique(np.round(pts, decimals=10), axis=0)
        n_pts = len(pts)

        # Always show individual vertex markers
        ternary_ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=config.marker_size, zorder=3)

        if n_pts == 1:
            continue

        if n_pts == 2:
            # Draw a line segment only
            ternary_ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=config.line_width)
            continue

        hull_pts = _compute_convex_hull_vertices(pts)
        _draw_polygon(ternary_ax, hull_pts, color, config)


def plot_credal_set(
    data: ArrayCredalSet,
    *,
    title: str | None = None,
    labels: list[str] | None = None,
    config: PlotConfig | None = None,
    show: bool = False,
    gridlines: bool = True,
) -> Axes:
    """Plot an Array credal set on a ternary simplex.

    Dispatches to a type-specific renderer based on the concrete credal set type.
    Only supports 3-class credal sets. For batched inputs, each element in the
    batch is overlaid on the same axes with a distinct color.

    The following types are supported:

    - ``ArrayProbabilityIntervalsCredalSet``: drawn as a filled feasibility polygon.
    - ``ArrayDistanceBasedCredalSet``: drawn as a filled polygon (derived from
      lower/upper bounds) with the nominal distribution marked as a point.
    - ``ArrayConvexCredalSet`` / ``ArrayDiscreteCredalSet``: drawn as the convex hull
      of the vertices/members with scatter markers at each vertex.
    - ``ArraySingletonCredalSet``: drawn as a single scatter marker.

    Args:
        data: The credal set to plot. Must have ``num_classes == 3``.
        title: Title of the plot.
        labels: Corner labels for the three classes. Defaults to ``["C1", "C2", "C3"]``.
        config: Plot configuration. Defaults to ``PlotConfig()``.
        show: Whether to call ``plt.show()`` after plotting.
        gridlines: Whether to display ternary gridlines. Defaults to ``True``.

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If ``num_classes != 3`` or labels length doesn't match.
        NotImplementedError: If ``data`` is not a supported Array credal set type.
    """
    num_classes = _get_num_classes(data)
    if num_classes != _NUM_TERNARY_CLASSES:
        msg = f"plot_credal_set only supports 3-class credal sets, got {num_classes} classes."
        raise ValueError(msg)

    config = config or PlotConfig()
    labels = labels or ["C1", "C2", "C3"]

    if len(labels) != _NUM_TERNARY_CLASSES:
        msg = f"Expected 3 labels, got {len(labels)}."
        raise ValueError(msg)

    fig = plt.figure(figsize=config.figure_size, dpi=config.dpi)
    ternary_ax = cast("TernaryAxes", fig.add_subplot(projection="ternary"))

    _draw_credal_set(data, ternary_ax, config)

    if gridlines:
        ternary_ax.grid(True, color=config.color_gridline)

    ternary_ax.set_tlabel(labels[0], fontsize=config.label_fontsize)
    ternary_ax.set_llabel(labels[1], fontsize=config.label_fontsize)
    ternary_ax.set_rlabel(labels[2], fontsize=config.label_fontsize)

    if title is not None:
        ternary_ax.set_title(title, fontsize=config.title_fontsize)

    if show:
        plt.show()

    return ternary_ax
