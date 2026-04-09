"""Main public entry point for credal set plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import mpltern  # noqa: F401  # must be imported for ternary projection support
import numpy as np

from probly.plot.config import PlotConfig

from ._binary import _draw_credal_set_binary, _setup_binary_axes
from ._radar_axes import _get_radar_axes
from ._spider import _draw_credal_set_spider, _setup_spider_axes
from ._ternary import _draw_credal_set_ternary

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpltern import TernaryAxes

    from probly.representation.credal_set.array import ArrayCategoricalCredalSet, ArraySingletonCredalSet

__all__ = ["plot_credal_set"]

_NUM_BINARY_CLASSES = 2
_NUM_TERNARY_CLASSES = 3
_MIN_SPIDER_CLASSES = 4

_OVERLAY_GT_COLOR = "#2c3e50"


def _draw_overlay_binary(
    ax: Axes,
    config: PlotConfig,
    ground_truth: ArraySingletonCredalSet | None,
) -> None:
    """Draw ground_truth overlay on binary axes."""
    if ground_truth is None:
        return
    gt = ground_truth.reshape(-1).array.probabilities
    for idx in range(gt.shape[0]):
        ax.scatter(
            gt[idx, 1],
            0,
            color=_OVERLAY_GT_COLOR,
            s=config.marker_size * 1.5,
            marker="*",
            zorder=5,
            label="Ground truth" if idx == 0 else None,
        )


def _draw_overlay_ternary(
    ax: Axes,
    config: PlotConfig,
    ground_truth: ArraySingletonCredalSet | None,
) -> None:
    """Draw ground_truth overlay on ternary axes."""
    if ground_truth is None:
        return
    ternary_ax = cast("TernaryAxes", ax)
    gt = ground_truth.reshape(-1).array.probabilities
    for idx in range(gt.shape[0]):
        # Slices (not scalars) are needed for mpltern's scatter API
        ternary_ax.scatter(
            gt[idx, 0:1],
            gt[idx, 1:2],
            gt[idx, 2:3],
            color=_OVERLAY_GT_COLOR,
            s=config.marker_size * 1.5,
            marker="*",
            zorder=5,
            label="Ground truth" if idx == 0 else None,
        )


def _draw_overlay_spider(
    ax: Axes,
    theta: np.ndarray,
    config: PlotConfig,
    ground_truth: ArraySingletonCredalSet | None,
) -> None:
    """Draw ground_truth overlay on spider axes."""
    if ground_truth is None:
        return
    gt = ground_truth.reshape(-1).array.probabilities
    for idx in range(gt.shape[0]):
        values = gt[idx]
        ax.plot(
            theta,
            values,
            color=_OVERLAY_GT_COLOR,
            linewidth=config.line_width,
            linestyle="-",
            zorder=5,
            label="Ground truth" if idx == 0 else None,
        )
        ax.scatter(
            theta,
            values,
            color=_OVERLAY_GT_COLOR,
            s=config.marker_size * 1.5,
            marker="*",
            zorder=6,
        )


def _has_legend(
    series_labels: list[str] | None,
    ground_truth: ArraySingletonCredalSet | None,
) -> bool:
    return series_labels is not None or ground_truth is not None


def _plot_binary(
    data: ArrayCategoricalCredalSet,
    labels: list[str],
    config: PlotConfig,
    series_labels: list[str] | None,
    title: str | None,
    ground_truth: ArraySingletonCredalSet | None,
) -> Axes:
    fig, ax = plt.subplots(figsize=(config.figure_size[0], 1.2), dpi=config.dpi)
    _setup_binary_axes(ax, labels, config)
    _draw_credal_set_binary(data, ax, config, series_labels)
    _draw_overlay_binary(ax, config, ground_truth)

    if _has_legend(series_labels, ground_truth):
        ax.legend()
    if title is not None:
        ax.set_title(title, fontsize=config.title_fontsize, pad=20)
    fig.tight_layout()
    return ax


def _plot_ternary(
    data: ArrayCategoricalCredalSet,
    labels: list[str],
    config: PlotConfig,
    series_labels: list[str] | None,
    title: str | None,
    gridlines: bool,
    ground_truth: ArraySingletonCredalSet | None,
) -> Axes:
    fig = plt.figure(figsize=config.figure_size, dpi=config.dpi)
    ternary_ax = cast("TernaryAxes", fig.add_subplot(projection="ternary"))

    _draw_credal_set_ternary(data, ternary_ax, config, series_labels)
    _draw_overlay_ternary(ternary_ax, config, ground_truth)

    if gridlines:
        ternary_ax.grid(True, color=config.color_gridline)

    ternary_ax.set_tlabel(labels[0], fontsize=config.label_fontsize)
    ternary_ax.set_llabel(labels[1], fontsize=config.label_fontsize)
    ternary_ax.set_rlabel(labels[2], fontsize=config.label_fontsize)

    ternary_ax.taxis.set_label_position("side")
    ternary_ax.laxis.set_label_position("side")
    ternary_ax.raxis.set_label_position("side")

    if _has_legend(series_labels, ground_truth):
        ternary_ax.legend()
    if title is not None:
        ternary_ax.set_title(title, fontsize=config.title_fontsize)
    return ternary_ax


def _plot_spider(
    data: ArrayCategoricalCredalSet,
    labels: list[str],
    config: PlotConfig,
    series_labels: list[str] | None,
    title: str | None,
    gridlines: bool,
    ground_truth: ArraySingletonCredalSet | None,
) -> Axes:
    radar_cls = _get_radar_axes(data.num_classes)
    fig = plt.figure(figsize=config.figure_size, dpi=config.dpi)
    ax = fig.add_subplot(projection=radar_cls.name)

    _setup_spider_axes(ax, labels, config, gridlines=gridlines)
    _draw_credal_set_spider(data, ax, config, series_labels)

    theta = np.linspace(0, 2 * np.pi, data.num_classes, endpoint=False)
    _draw_overlay_spider(ax, theta, config, ground_truth)

    if _has_legend(series_labels, ground_truth):
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if title is not None:
        ax.set_title(title, fontsize=config.title_fontsize, pad=20)
    fig.tight_layout()
    return ax


def plot_credal_set(
    data: ArrayCategoricalCredalSet,
    *,
    title: str | None = None,
    labels: list[str] | None = None,
    series_labels: list[str] | None = None,
    config: PlotConfig | None = None,
    show: bool = False,
    gridlines: bool = True,
    ground_truth: ArraySingletonCredalSet | None = None,
) -> Axes:
    """Plot an Array credal set.

    For 2-class credal sets, renders a horizontal interval plot where the
    x-axis represents P(class 2). For 3-class credal sets, renders a ternary
    simplex diagram. For 4+ classes, renders a spider (radar) plot.

    Dispatches to a type-specific renderer based on the concrete credal set type.
    For batched inputs, each element in the batch is overlaid on the same axes
    with a distinct color.

    The following types are supported:

    - :class:`~probly.representation.credal_set.array.ArrayProbabilityIntervalsCredalSet`:
      drawn as a filled interval (2-class), feasibility envelope (3-class), or
      constant-width bars on spokes (4+ classes).
    - :class:`~probly.representation.credal_set.array.ArrayDistanceBasedCredalSet`:
      drawn as a filled interval/envelope/bars with the nominal distribution
      marked as a point.
    - :class:`~probly.representation.credal_set.array.ArrayConvexCredalSet` /
      :class:`~probly.representation.credal_set.array.ArrayDiscreteCredalSet`:
      drawn as an interval with vertex markers (2-class), convex hull envelope
      (3-class), or min/max envelope (4+ classes).
    - :class:`~probly.representation.credal_set.array.ArraySingletonCredalSet`:
      drawn as a single scatter marker or closed envelope on spokes.

    Args:
        data: The credal set to plot.
        title: Title of the plot.
        labels: Class labels. Defaults to ``["C0", "C1", ...]``.
        series_labels: Optional legend labels, one per batch element. When
            provided a legend is shown. Defaults to ``None`` (no legend).
        config: Plot configuration. Defaults to ``PlotConfig()``.
        show: Whether to call ``plt.show()`` after plotting.
        gridlines: Whether to display gridlines. Used for 3-class (ternary) and
            4+-class (spider) plots. Defaults to ``True``.
        ground_truth: Optional ground-truth distribution to overlay as a star
            marker. Must be an
            :class:`~probly.representation.credal_set.array.ArraySingletonCredalSet`
            with the same number of classes as ``data``.

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If labels length does not match the number of classes.
        NotImplementedError: If ``data`` is not a supported Array credal set type.
    """
    num_classes = data.num_classes
    config = config or PlotConfig()

    if num_classes == _NUM_BINARY_CLASSES:
        labels = labels or ["C0", "C1"]
        if len(labels) != _NUM_BINARY_CLASSES:
            msg = f"Expected 2 labels, got {len(labels)}."
            raise ValueError(msg)
        ax = _plot_binary(data, labels, config, series_labels, title, ground_truth)

    elif num_classes == _NUM_TERNARY_CLASSES:
        labels = labels or ["C0", "C1", "C2"]
        if len(labels) != _NUM_TERNARY_CLASSES:
            msg = f"Expected 3 labels, got {len(labels)}."
            raise ValueError(msg)
        ax = _plot_ternary(data, labels, config, series_labels, title, gridlines, ground_truth)

    elif num_classes >= _MIN_SPIDER_CLASSES:
        labels = labels or [f"C{i}" for i in range(num_classes)]
        if len(labels) != num_classes:
            msg = f"Expected {num_classes} labels, got {len(labels)}."
            raise ValueError(msg)
        ax = _plot_spider(
            data,
            labels,
            config,
            series_labels,
            title,
            gridlines,
            ground_truth,
        )

    else:
        msg = f"plot_credal_set requires at least 2 classes, got {num_classes}."
        raise ValueError(msg)

    if show:
        plt.show()

    return ax
