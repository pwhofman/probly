"""Ternary (3-class) credal set simplex drawing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from flextype import flexdispatch
from probly.representation.credal_set.array import (
    ArrayCategoricalCredalSet,
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

from ._geometry import _compute_convex_hull_vertices, _compute_interval_vertices

if TYPE_CHECKING:
    from mpltern import TernaryAxes

    from probly.plot.config import PlotConfig

_NUM_TERNARY_CLASSES = 3


def _draw_polygon(
    ternary_ax: TernaryAxes,
    vertices: np.ndarray,
    color: object,
    config: PlotConfig,
    *,
    label: str | None = None,
) -> None:
    """Render a filled polygon and its outline on ternary axes.

    Args:
        ternary_ax: The mpltern ternary axes to draw on.
        vertices: Array of shape (n, 3) with polygon vertices in winding order.
        color: Matplotlib color for both fill and outline.
        config: Plot configuration supplying fill_alpha and line_width.
        label: Optional legend label for this polygon.
    """
    closed = np.vstack([vertices, vertices[0]])
    ternary_ax.fill(closed[:, 0], closed[:, 1], closed[:, 2], alpha=config.fill_alpha, color=color, label=label)
    ternary_ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, linewidth=config.line_width)


@flexdispatch
def _draw_credal_set_ternary(
    data: ArrayCategoricalCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    """Draw a credal set on ternary axes.

    Args:
        data: The credal set to draw.
        ternary_ax: The mpltern ternary axes to draw on.
        config: Plot configuration.
        series_labels: Optional per-series legend labels.

    Raises:
        NotImplementedError: If no handler is registered for the given type.
    """
    msg = f"Unsupported credal set type: {type(data).__name__}"
    raise NotImplementedError(msg)


@_draw_credal_set_ternary.register(ArraySingletonCredalSet)
def _draw_singleton(
    data: ArraySingletonCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    data = data.reshape(-1)
    pts = data.array.unnormalized_probabilities
    n_sets = pts.shape[0]

    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        p = pts[idx]
        ternary_ax.scatter(p[0:1], p[1:2], p[2:3], color=color, s=config.marker_size, zorder=3, label=label)


@_draw_credal_set_ternary.register(ArrayProbabilityIntervalsCredalSet)
def _draw_intervals(
    data: ArrayProbabilityIntervalsCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    lower_all = data.lower_bounds.reshape(-1, _NUM_TERNARY_CLASSES)
    upper_all = data.upper_bounds.reshape(-1, _NUM_TERNARY_CLASSES)
    n_sets = lower_all.shape[0]

    for idx in range(n_sets):
        vertices = _compute_interval_vertices(lower_all[idx], upper_all[idx])
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None

        unique_verts = np.unique(np.round(vertices, decimals=8), axis=0)
        if len(unique_verts) <= 1:
            # Zero-area: all vertices collapse to a single point; draw a dot
            pt = unique_verts[0] if len(unique_verts) == 1 else vertices[0]
            ternary_ax.scatter(pt[0:1], pt[1:2], pt[2:3], color=color, s=config.marker_size, zorder=3, label=label)
            continue

        _draw_polygon(ternary_ax, vertices, color, config, label=label)


@_draw_credal_set_ternary.register(ArrayDistanceBasedCredalSet)
def _draw_distance_based(
    data: ArrayDistanceBasedCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    lower_all = data.lower().reshape(-1, _NUM_TERNARY_CLASSES)
    upper_all = data.upper().reshape(-1, _NUM_TERNARY_CLASSES)
    nominal_all = data.nominal.unnormalized_probabilities.reshape(-1, _NUM_TERNARY_CLASSES)
    n_sets = lower_all.shape[0]

    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        vertices = _compute_interval_vertices(lower_all[idx], upper_all[idx])

        unique_verts = np.unique(np.round(vertices, decimals=8), axis=0)
        if len(unique_verts) <= 1:
            pt = unique_verts[0] if len(unique_verts) == 1 else vertices[0]
            ternary_ax.scatter(pt[0:1], pt[1:2], pt[2:3], color=color, s=config.marker_size, zorder=3, label=label)
        else:
            _draw_polygon(ternary_ax, vertices, color, config, label=label)

        nom = nominal_all[idx]
        ternary_ax.scatter(nom[0:1], nom[1:2], nom[2:3], color=color, s=config.marker_size, zorder=3)


@_draw_credal_set_ternary.register(ArrayDiscreteCredalSet)
def _draw_discrete_set(
    data: ArrayDiscreteCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    arr = data.reshape(-1).array.unnormalized_probabilities
    n_sets = arr.shape[0]

    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        pts = arr[idx]
        ternary_ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=config.marker_size, zorder=3, label=label)


@_draw_credal_set_ternary.register(ArrayConvexCredalSet)
def _draw_convex_set(
    data: ArrayConvexCredalSet,
    ternary_ax: TernaryAxes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    arr = data.reshape(-1).array.unnormalized_probabilities
    n_sets = arr.shape[0]

    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        pts = arr[idx]

        pts = np.unique(np.round(pts, decimals=10), axis=0)
        n_pts = len(pts)

        if n_pts == 1:
            ternary_ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=config.marker_size, zorder=3, label=label
            )
            continue

        if n_pts == 2:
            ternary_ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=config.marker_size, zorder=3)
            ternary_ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=config.line_width, label=label)
            continue

        ternary_ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=config.marker_size, zorder=3)
        hull_pts = _compute_convex_hull_vertices(pts)
        _draw_polygon(ternary_ax, hull_pts, color, config, label=label)
