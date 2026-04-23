"""Binary (2-class) credal set interval drawing."""

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

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from probly.plot.config import PlotConfig

_NUM_BINARY_CLASSES = 2
_BINARY_Y_HEIGHT = 0.05
_BINARY_Y_PAD = 0.02


def _setup_binary_axes(
    ax: Axes,
    labels: list[str],  # noqa: ARG001
    config: PlotConfig,
) -> None:
    """Set up axes for a binary credal set interval plot.

    Uses a native matplotlib x-axis with ticks from 0 to 1 and an axis label
    of "Probability of Class 1". The y-axis and all spines except the bottom
    are hidden.

    Args:
        ax: The matplotlib axes to configure.
        labels: Two class labels (kept for API compatibility).
        config: Plot configuration.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.set_xlabel("Probability of Class 1", fontsize=config.label_fontsize)

    ax.set_ylim(-_BINARY_Y_PAD, _BINARY_Y_HEIGHT + _BINARY_Y_PAD)
    ax.set_yticks([])

    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


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
    y_extent = np.array([0.0, _BINARY_Y_HEIGHT])
    ax.fill_betweenx(y_extent, low, high, color=color, alpha=config.fill_alpha, zorder=2, label=label)


@flexdispatch
def _draw_credal_set_binary(
    data: ArrayCategoricalCredalSet,
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
    data = data.reshape(-1)
    arr = data.array.unnormalized_probabilities
    n_sets = arr.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        ax.scatter(arr[idx, 1], 0, color=color, s=config.marker_size, zorder=3, label=label)


@_draw_credal_set_binary.register(ArrayProbabilityIntervalsCredalSet)
def _draw_intervals_binary(
    data: ArrayProbabilityIntervalsCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    data = data.reshape(-1)
    lower_all = data.lower_bounds
    upper_all = data.upper_bounds
    n_sets = lower_all.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        low, high = lower_all[idx, 1], upper_all[idx, 1]
        if np.isclose(low, high):
            ax.scatter(low, 0, color=color, s=config.marker_size, zorder=3, label=label)
        else:
            _draw_binary_interval(ax, low, high, color, config, label=label)


@_draw_credal_set_binary.register(ArrayDistanceBasedCredalSet)
def _draw_distance_based_binary(
    data: ArrayDistanceBasedCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    data = data.reshape(-1)
    lower_all = data.lower()
    upper_all = data.upper()
    nominal_all = data.nominal.unnormalized_probabilities
    n_sets = lower_all.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        _draw_binary_interval(ax, lower_all[idx, 1], upper_all[idx, 1], color, config, label=label)
        ax.scatter(nominal_all[idx, 1], 0, color=color, s=config.marker_size, zorder=3)


@_draw_credal_set_binary.register(ArrayDiscreteCredalSet | ArrayConvexCredalSet)
def _draw_vertex_set_binary(
    data: ArrayConvexCredalSet | ArrayDiscreteCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None = None,
) -> None:
    arr = data.reshape(-1).array.unnormalized_probabilities
    n_sets = arr.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        p2_values = arr[idx, :, 1]
        _draw_binary_interval(ax, float(p2_values.min()), float(p2_values.max()), color, config, label=label)
        ax.scatter(p2_values, np.zeros_like(p2_values), color=color, s=config.marker_size, zorder=3)
