"""Minimal plot styling and the shared ranked-bar renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np

from paper_plots.metadata import BASE_VARIANT_KEYS, CREDAL_METHOD_KEYS, DIRECT_METHOD_KEYS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

_FONT_FAMILY = "Fira Sans"


def _apply_rc_defaults() -> None:
    """Set global matplotlib rcParams for the Fira Sans font family."""
    rc = mpl.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = [_FONT_FAMILY]
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10
    rc["axes.labelsize"] = 12
    rc["axes.labelweight"] = 600
    rc["axes.titlesize"] = 14
    rc["axes.titleweight"] = 700
    rc["font.weight"] = 300
    rc["legend.fontsize"] = 10
    rc["figure.titlesize"] = 16
    rc["figure.titleweight"] = 700
    # NeurIPS PDFs may only contain Type 1 or embedded TrueType fonts, not matplotlib's default Type 3.
    rc["pdf.fonttype"] = 42
    rc["mathtext.fontset"] = "custom"
    rc["mathtext.rm"] = _FONT_FAMILY
    rc["mathtext.it"] = f"{_FONT_FAMILY}:italic"
    rc["mathtext.bf"] = f"{_FONT_FAMILY}:bold"
    rc["mathtext.sf"] = _FONT_FAMILY
    rc["mathtext.tt"] = _FONT_FAMILY


_apply_rc_defaults()


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Shared style knobs for the bar plots."""

    categorical_palette: tuple[str, ...] = (
        "#1e88e5",  # blue
        "#ff0d57",  # pink-red
        "#2ecc71",  # green
        "#f39c12",  # orange
        "#9b59b6",  # purple
        "#16a085",  # teal
        "#d35400",  # dark orange
        "#34495e",  # slate
        "#1abc9c",  # turquoise
        "#e74c3c",  # alizarin red
        "#f1c40f",  # sun yellow
        "#00bcd4",  # cyan
        "#795548",  # brown
        "#cddc39",  # lime
        "#6610f2",  # indigo
    )
    color_gridline: str = "#e8e8e8"
    color_whisker: str = "#333333"  # std whiskers of bar plots
    color_negative: str = "#1e88e5"  # ID histograms (blue)
    color_neutral: str = "#7f8c8d"  # neutral grey (e.g. base curve in sp_arc)
    color_positive: str = "#ff0d57"  # OOD histograms / SP & OOD baseline reference line (pink-red)
    grid_alpha: float = 0.5
    grid_linestyle: str = "--"
    line_width: float = 1.5
    fill_alpha: float = 0.3
    histogram_alpha: float = 0.6

    def color(self, index: int) -> str:
        """Return a color from the categorical palette (wrap-around)."""
        return self.categorical_palette[index % len(self.categorical_palette)]


_BASE_BLUE = "#1e88e5"  # PlotConfig.color(0); used as the anchor for tiered shades


def _lighten(color: str, mix: float) -> tuple[float, float, float]:
    """Mix ``color`` with white. ``mix=0`` keeps it, ``mix=1`` returns white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * mix, g + (1 - g) * mix, b + (1 - b) * mix)


_DIRECT_BLUE = "#0b3c7d"
_CREDAL_BLUE = _lighten(_BASE_BLUE, 0.40)
_CALIBRATION_BLUE = _lighten(_BASE_BLUE, 0.78)
_FAMILY_LEGEND: tuple[tuple[str, str | tuple[float, float, float]], ...] = (
    ("Direct", _DIRECT_BLUE),
    ("2nd-order Distr.", _BASE_BLUE),
    ("Credal", _CREDAL_BLUE),
    ("Calibration", _CALIBRATION_BLUE),
)
# Space below the lowest and above the highest drawn element, as a fraction of their distance.
YLIM_BOTTOM_PAD = 0.08
YLIM_TOP_PAD = 0.03
# Whiskers go in front of bars, lines and markers, so none of them can hide one.
WHISKER_ZORDER = 10
# Half the width of a whisker's end bars, in x-axis units (one unit is one bar slot).
WHISKER_CAP_HALF_WIDTH = 0.2
# Radius of the rounded top corners of bars, in points at full figure size.
BAR_CORNER_RADIUS_PT = 3.0
# Points between the top of the axes and the title; the legend row sits in this gap.
LEGEND_ROW_PAD = 22


def bar_color_for(method: str | None) -> str | tuple[float, float, float]:
    """Per-method bar color for the SP / OOD bar plots, one blue per method family.

    Calibration variants of ``base`` get the lightest blue, credal-set methods the
    next one, direct methods the darkest, and everything else counts as a
    second-order distribution method and gets the default blue.
    """
    if method is None:
        return _BASE_BLUE
    if method in BASE_VARIANT_KEYS:
        return _CALIBRATION_BLUE
    if method in CREDAL_METHOD_KEYS:
        return _CREDAL_BLUE
    if method in DIRECT_METHOD_KEYS:
        return _DIRECT_BLUE
    return _BASE_BLUE


def _draw_baseline(ax: Axes, baseline: dict, cfg: PlotConfig) -> Line2D:
    """Draw the baseline as a dashed line with a shaded band of one std on either side."""
    mean, std = baseline["mean"], baseline["std"]
    ax.axhspan(mean - std, mean + std, color=cfg.color_positive, alpha=0.22, linewidth=0, zorder=1.4)
    return ax.axhline(
        mean,
        color=cfg.color_positive,
        linestyle="--",
        linewidth=1.4,
        zorder=1.5,
        label=f"base ({mean:.3f} $\\pm$ {std:.1g})",
    )


def fit_ylim(ax: Axes, values: Iterable[float]) -> None:
    """Fit the y-axis to everything drawn, so a plot spends its height on the data.

    Args:
        ax: Axes to adjust.
        values: Heights of all drawn elements: bar tops, whisker ends, markers, reference lines.
    """
    heights = list(values)
    low, high = min(heights), max(heights)
    span = high - low
    # More room below than above: the shortest bar must still read as a bar.
    ax.set_ylim(max(0.0, low - YLIM_BOTTOM_PAD * span), high + YLIM_TOP_PAD * span)


def draw_whiskers(
    ax: Axes,
    x: Iterable[float],
    means: Sequence[float],
    stds: Sequence[float],
    *,
    linewidth: float = 1.2,
) -> None:
    """Draw ``mean +- std`` whiskers with rounded line ends, in front of everything else.

    Args:
        ax: Axes to draw on.
        x: Bar positions.
        means: Whisker centers.
        stds: Whisker half lengths.
        linewidth: Line width in points.
    """
    cap = WHISKER_CAP_HALF_WIDTH
    whiskers = [
        [
            (xi - cap, mean + std),
            (xi + cap, mean + std),
            (xi, mean + std),
            (xi, mean - std),
            (xi - cap, mean - std),
            (xi + cap, mean - std),
        ]
        for xi, mean, std in zip(x, means, stds, strict=True)
    ]
    collection = LineCollection(
        whiskers,
        colors=PlotConfig().color_whisker,
        linewidths=linewidth,
        capstyle="round",
        joinstyle="round",
        zorder=WHISKER_ZORDER,
    )
    ax.add_collection(collection)
    ax.autoscale_view()


def round_bar_tops(ax: Axes, bars: Iterable[Rectangle], *, radius_pt: float = BAR_CORNER_RADIUS_PT) -> None:
    """Replace ``bars`` with shapes of the same size whose top two corners are rounded.

    The shapes live in data coordinates, so the radius is converted using the current
    axes size and limits: call this after the limits are final and after ``tight_layout``,
    or the corners stretch. The limits are frozen here so the new shapes cannot move them.

    Args:
        ax: Axes holding the bars.
        bars: Bars to replace, as returned by ``ax.bar``.
        radius_pt: Corner radius in points, capped at half the bar width and at its height.
    """
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    pixels_per_point = ax.figure.dpi / 72
    origin = ax.transData.inverted().transform((0, 0))
    one_point = ax.transData.inverted().transform((pixels_per_point, pixels_per_point))
    x_per_point, y_per_point = np.abs(one_point - origin)
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    for bar in list(bars):
        left, bottom = bar.get_x(), bar.get_y()
        right, top = left + bar.get_width(), bottom + bar.get_height()
        rx = min(radius_pt * x_per_point, bar.get_width() / 2)
        ry = min(radius_pt * y_per_point, bar.get_height())
        corners = [
            (left, bottom),
            (left, top - ry),
            (left, top),
            (left + rx, top),
            (right - rx, top),
            (right, top),
            (right, top - ry),
            (right, bottom),
            (left, bottom),
        ]
        ax.add_patch(
            PathPatch(
                Path(corners, codes),
                facecolor=bar.get_facecolor(),
                edgecolor="none",
                linewidth=0,
                zorder=bar.get_zorder(),
            ),
        )
        bar.remove()


def draw_legend_row(ax: Axes, handles: list[Artist], *, columnspacing: float = 1.4) -> None:
    """Draw ``handles`` as one legend row between the title and the axes.

    The title must be set with ``pad=LEGEND_ROW_PAD`` so the row has room. A row wider
    than the axes makes ``tight_layout`` shrink them, so keep the labels short.
    """
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        borderaxespad=0.2,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=columnspacing,
    )


def draw_bars(
    ranking: list[dict],
    *,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    figsize: tuple[float, float] | None = None,
    baseline: dict | None = None,
) -> Figure:
    """Render a descending bar chart from a pre-sorted ``ranking`` list.

    Args:
        ranking: List of ``{"label", "mean", "std"}`` dicts sorted descending.
        ylabel: Y-axis label.
        title: Bold main title.
        subtitle: Optional regular-weight inline subtitle.
        figsize: Optional explicit figure size in inches. When ``None`` an
            adaptive default is used that scales with bar count.
        baseline: Optional ``{"mean", "std"}`` dict of the reference method, drawn
            as a line with a std band.

    Returns:
        The matplotlib Figure.
    """
    cfg = PlotConfig()
    labels = [entry["label"] for entry in ranking]
    means = [entry["mean"] for entry in ranking]
    stds = [entry["std"] for entry in ranking]
    x = np.arange(len(labels))

    if figsize is None:
        figsize = (max(6.0, 0.6 * len(labels) + 2.0), 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.8
    bar_colors = [bar_color_for(entry.get("method")) for entry in ranking]
    bars = ax.bar(
        x,
        means,
        width=bar_width,
        color=bar_colors,
        edgecolor="none",
    )
    draw_whiskers(ax, x, means, stds)
    inter_bar_gap = 1.0 - bar_width
    ax.set_xlim(-bar_width / 2 - inter_bar_gap, len(labels) - 1 + bar_width / 2 + inter_bar_gap)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if subtitle:
        subtitle_math = subtitle.replace(" ", r"\ ")
        ax.set_title(f"{title}  $\\mathrm{{{subtitle_math}}}$", pad=LEGEND_ROW_PAD)
    else:
        ax.set_title(title, pad=LEGEND_ROW_PAD)
    ax.yaxis.grid(
        visible=True,
        linestyle=cfg.grid_linestyle,
        alpha=cfg.grid_alpha,
        color=cfg.color_gridline,
    )
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    extents = [m + sign * d for m, d in zip(means, stds, strict=True) for sign in (-1, 1)]
    if baseline is not None:
        extents += [baseline["mean"] - baseline["std"], baseline["mean"] + baseline["std"]]
    fit_ylim(ax, extents)
    handles: list[Artist] = [Patch(color=color, label=label) for label, color in _FAMILY_LEGEND]
    if baseline is not None:
        handles.append(_draw_baseline(ax, baseline, cfg))
    draw_legend_row(ax, handles)
    fig.tight_layout()
    round_bar_tops(ax, bars)
    return fig
