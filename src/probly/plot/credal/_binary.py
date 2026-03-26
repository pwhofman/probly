"""Binary (2-class) credal set interval drawing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lazy_dispatch import lazydispatch
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from probly.plot.config import PlotConfig

    type ArrayCredalSet = (
        ArrayProbabilityIntervalsCredalSet
        | ArrayDistanceBasedCredalSet
        | ArrayConvexCredalSet
        | ArrayDiscreteCredalSet
        | ArraySingletonCredalSet
    )

_NUM_BINARY_CLASSES = 2
_BINARY_Y_MARGIN = 0.1
_BINARY_TICK_LENGTH = 0.02
_BINARY_TICK_LABEL_OFFSET = -0.05


def _setup_binary_axes(
    ax: Axes,
    labels: list[str],
    config: PlotConfig,
) -> None:
    """Set up axes for a binary credal set interval plot.

    Draws a horizontal baseline from 0 to 1 with tick marks at 0.1 intervals
    and class labels at each endpoint.

    Args:
        ax: The matplotlib axes to configure.
        labels: Two class labels for the endpoints.
        config: Plot configuration.
    """
    ax.plot([0, 1], [0, 0], color="black", linewidth=config.line_width, zorder=0)

    tick_values = np.linspace(0.0, 1.0, 11)
    for t in tick_values[1:-1]:
        ax.plot(
            [t, t],
            [-_BINARY_TICK_LENGTH / 2, _BINARY_TICK_LENGTH / 2],
            color="black",
        )
        ax.text(
            t,
            _BINARY_TICK_LABEL_OFFSET,
            f"{t:.1f}",
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.text(0, _BINARY_TICK_LABEL_OFFSET - 0.05, labels[0], ha="center", va="top", fontsize=config.label_fontsize)
    ax.text(1, _BINARY_TICK_LABEL_OFFSET - 0.05, labels[1], ha="center", va="top", fontsize=config.label_fontsize)

    ax.axis("off")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.2)


def _draw_binary_interval(
    ax: Axes,
    low: float,
    high: float,
    color: object,
    config: PlotConfig,
    *,
    label: str | None = None,
) -> None:
    """Draw a shaded interval band on binary axes.

    Args:
        ax: The matplotlib axes to draw on.
        low: Lower bound of the interval on [0, 1].
        high: Upper bound of the interval on [0, 1].
        color: Matplotlib color for the band.
        config: Plot configuration.
        label: Optional legend label for this interval.
    """
    y_margin = np.array([_BINARY_Y_MARGIN, -_BINARY_Y_MARGIN])
    ax.fill_betweenx(y_margin, low, high, color=color, alpha=config.fill_alpha, zorder=2, label=label)


@lazydispatch
def _draw_credal_set_binary(
    data: ArrayCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    """Draw a binary credal set on standard axes.

    Args:
        data: The credal set to draw.
        ax: The matplotlib axes to draw on.
        config: Plot configuration.
        series_labels: Optional per-series legend labels.

    Raises:
        NotImplementedError: If no handler is registered for the given type.
    """
    msg = f"Unsupported credal set type: {type(data).__name__}"
    raise NotImplementedError(msg)


@_draw_credal_set_binary.register(ArraySingletonCredalSet)
def _draw_singleton_binary(
    data: ArraySingletonCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    arr = data.array.reshape(-1, _NUM_BINARY_CLASSES)
    n_sets = arr.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels else None
        ax.scatter(arr[idx, 1], 0, color=color, s=config.marker_size, zorder=3, label=label)


@_draw_credal_set_binary.register(ArrayProbabilityIntervalsCredalSet)
def _draw_intervals_binary(
    data: ArrayProbabilityIntervalsCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    lower_all = data.lower_bounds.reshape(-1, _NUM_BINARY_CLASSES)
    upper_all = data.upper_bounds.reshape(-1, _NUM_BINARY_CLASSES)
    n_sets = lower_all.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels else None
        _draw_binary_interval(ax, lower_all[idx, 1], upper_all[idx, 1], color, config, label=label)


@_draw_credal_set_binary.register(ArrayDistanceBasedCredalSet)
def _draw_distance_based_binary(
    data: ArrayDistanceBasedCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    lower_all = data.lower().reshape(-1, _NUM_BINARY_CLASSES)
    upper_all = data.upper().reshape(-1, _NUM_BINARY_CLASSES)
    nominal_all = data.nominal.reshape(-1, _NUM_BINARY_CLASSES)
    n_sets = lower_all.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels else None
        _draw_binary_interval(ax, lower_all[idx, 1], upper_all[idx, 1], color, config, label=label)
        ax.scatter(nominal_all[idx, 1], 0, color=color, s=config.marker_size, zorder=3)


@_draw_credal_set_binary.register(ArrayDiscreteCredalSet | ArrayConvexCredalSet)
def _draw_vertex_set_binary(
    data: ArrayConvexCredalSet | ArrayDiscreteCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    arr = data.array.reshape(-1, data.array.shape[-2], _NUM_BINARY_CLASSES)
    n_sets = arr.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels else None
        p2_values = arr[idx, :, 1]
        _draw_binary_interval(ax, float(p2_values.min()), float(p2_values.max()), color, config, label=label)
        ax.scatter(p2_values, np.zeros_like(p2_values), color=color, s=config.marker_size, zorder=3)
