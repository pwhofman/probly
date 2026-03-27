"""Main public entry point for credal set plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import mpltern  # noqa: F401

from probly.plot.config import PlotConfig

from ._binary import _draw_credal_set_binary, _setup_binary_axes
from ._dispatch import _get_num_classes
from ._ternary import _draw_credal_set_ternary

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpltern import TernaryAxes

    from probly.representation.credal_set.array import (
        ArrayConvexCredalSet,
        ArrayDiscreteCredalSet,
        ArrayDistanceBasedCredalSet,
        ArrayProbabilityIntervalsCredalSet,
        ArraySingletonCredalSet,
    )

    type ArrayCredalSet = (
        ArrayProbabilityIntervalsCredalSet
        | ArrayDistanceBasedCredalSet
        | ArrayConvexCredalSet
        | ArrayDiscreteCredalSet
        | ArraySingletonCredalSet
    )

__all__ = ["plot_credal_set"]

_NUM_BINARY_CLASSES = 2
_NUM_TERNARY_CLASSES = 3


def plot_credal_set(
    data: ArrayCredalSet,
    *,
    title: str | None = None,
    labels: list[str] | None = None,
    series_labels: list[str] | None = None,
    config: PlotConfig | None = None,
    show: bool = False,
    gridlines: bool = True,
) -> Axes:
    """Plot an Array credal set.

    For 2-class credal sets, renders a horizontal interval plot where the
    x-axis represents P(class 2). For 3-class credal sets, renders a ternary
    simplex diagram.

    Dispatches to a type-specific renderer based on the concrete credal set type.
    For batched inputs, each element in the batch is overlaid on the same axes
    with a distinct color.

    The following types are supported:

    - :class:`~probly.representation.credal_set.array.ArrayProbabilityIntervalsCredalSet`:
      drawn as a filled interval (2-class) or feasibility polygon (3-class).
    - :class:`~probly.representation.credal_set.array.ArrayDistanceBasedCredalSet`:
      drawn as a filled interval/polygon with the nominal distribution marked as a point.
    - :class:`~probly.representation.credal_set.array.ArrayConvexCredalSet` /
      :class:`~probly.representation.credal_set.array.ArrayDiscreteCredalSet`:
      drawn as an interval with vertex markers (2-class) or convex hull polygon (3-class).
    - :class:`~probly.representation.credal_set.array.ArraySingletonCredalSet`:
      drawn as a single scatter marker.

    Args:
        data: The credal set to plot. Must have 2 or 3 classes.
        title: Title of the plot.
        labels: Class labels. Defaults to ``["C0", "C1"]`` for binary or
            ``["C0", "C1", "C2"]`` for ternary.
        series_labels: Optional legend labels, one per batch element. When
            provided a legend is shown. Defaults to ``None`` (no legend).
        config: Plot configuration. Defaults to ``PlotConfig()``.
        show: Whether to call ``plt.show()`` after plotting.
        gridlines: Whether to display ternary gridlines. Only used for 3-class
            plots. Defaults to ``True``.

    Returns:
        The matplotlib axes with the plot.

    Raises:
        ValueError: If ``num_classes`` is not 2 or 3, or labels length doesn't match.
        NotImplementedError: If ``data`` is not a supported Array credal set type.
    """
    num_classes = _get_num_classes(data)
    config = config or PlotConfig()

    if num_classes == _NUM_BINARY_CLASSES:
        labels = labels or ["C0", "C1"]
        if len(labels) != _NUM_BINARY_CLASSES:
            msg = f"Expected 2 labels, got {len(labels)}."
            raise ValueError(msg)

        fig, ax = plt.subplots(figsize=(config.figure_size[0], 1.2), dpi=config.dpi)
        _setup_binary_axes(ax, labels, config)
        _draw_credal_set_binary(data, ax, config, series_labels)

        if series_labels is not None:
            ax.legend()

        fig.tight_layout()

        if title is not None:
            fig.suptitle(title, fontsize=config.title_fontsize, y=1.05)

        if show:
            plt.show()

        return ax

    if num_classes == _NUM_TERNARY_CLASSES:
        labels = labels or ["C0", "C1", "C2"]
        if len(labels) != _NUM_TERNARY_CLASSES:
            msg = f"Expected 3 labels, got {len(labels)}."
            raise ValueError(msg)

        fig = plt.figure(figsize=config.figure_size, dpi=config.dpi)
        ternary_ax = cast("TernaryAxes", fig.add_subplot(projection="ternary"))

        _draw_credal_set_ternary(data, ternary_ax, config, series_labels)

        if gridlines:
            ternary_ax.grid(True, color=config.color_gridline)

        ternary_ax.set_tlabel(labels[0], fontsize=config.label_fontsize)
        ternary_ax.set_llabel(labels[1], fontsize=config.label_fontsize)
        ternary_ax.set_rlabel(labels[2], fontsize=config.label_fontsize)

        ternary_ax.taxis.set_label_position("side")
        ternary_ax.laxis.set_label_position("side")
        ternary_ax.raxis.set_label_position("side")

        if series_labels is not None:
            ternary_ax.legend()

        if title is not None:
            ternary_ax.set_title(title, fontsize=config.title_fontsize)

        if show:
            plt.show()

        return ternary_ax

    msg = f"plot_credal_set supports 2- or 3-class credal sets, got {num_classes} classes."
    raise ValueError(msg)
