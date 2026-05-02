"""Plot one stream's per-step trajectories from a ``run_stream.py`` JSON.

Produces two PDFs per stream:

* ``<stream>_decomposition.pdf`` — total / aleatoric / epistemic over time.
* ``<stream>_accuracy.pdf`` — epistemic + rolling accuracy on twin axes.

Reuses :class:`probly.plot.PlotConfig` for colors, fonts, and styling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from probly.plot import PlotConfig

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT_SMOOTH_WIN: Final[int] = 75
_ACC_SMOOTH_FACTOR: Final[float] = 2.0
DEFAULT_FIGSIZE: Final[tuple[float, float]] = (7.0, 4.0)
_BOTTOM_MARGIN: Final[float] = 0.30
_BOTTOM_MARGIN_NO_LEGEND: Final[float] = 0.16
_LEGEND_FIGSIZE: Final[tuple[float, float]] = (7.0, 0.55)
_YLIM_WARMUP_T: Final[int] = 100  # ignore first samples when picking uncertainty y-limit
_YLIM_HEADROOM: Final[float] = 1.10
_ACC_COLOR: Final[str] = "#a3aeaf"  # one step lighter than PlotConfig's color_neutral
_ACC_AXIS_COLOR: Final[str] = "#7f8c8d"  # darker gray for right-axis ticks / label
_FONT_SCALE: Final[float] = 1.3
for _rc_key in (
    "xtick.labelsize",
    "ytick.labelsize",
    "axes.labelsize",
    "axes.titlesize",
    "legend.fontsize",
    "figure.titlesize",
):
    mpl.rcParams[_rc_key] = float(mpl.rcParams[_rc_key]) * _FONT_SCALE


def _rolling(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=1).mean()


def _aggregate(records: pd.DataFrame, col: str, win: int) -> pd.DataFrame:
    """Per-seed rolling mean, then per-step median + IQR across seeds."""
    return (
        records.assign(rolled=lambda d: d.groupby("seed")[col].transform(_rolling, win=win))
        .groupby("t")["rolled"]
        .agg(med="median", lo=lambda s: s.quantile(0.25), hi=lambda s: s.quantile(0.75))
        .reset_index()
    )


def _uncertainty_ylim_top(records: pd.DataFrame, cols: tuple[str, ...], smooth_win: int) -> float:
    """Pick the upper y-limit from the median trace, ignoring the warm-up region.

    The first :data:`_YLIM_WARMUP_T` samples often have a spurious AU/TU spike
    while the trees haven't grown enough to give a meaningful entropy estimate,
    which would otherwise dominate the y-axis and squash the rest of the run.
    Uses the median (not the upper IQR) so the plot scales to the typical
    behaviour rather than seed-tail variation.
    """
    cap = 0.0
    for col in cols:
        agg = _aggregate(records, col, smooth_win)
        post = agg[agg["t"] >= _YLIM_WARMUP_T]
        if not post.empty:
            cap = max(cap, float(post["med"].max()))
    return cap * _YLIM_HEADROOM if cap > 0 else 1.0


_DRIFT_LABEL = "concept drift"
_DRIFT_LABEL_Y = 0.12  # axes-fraction height for the drift annotation (default)
_FIGURE_TITLE = r"$\mathtt{river}$ Adaptive Random Forest classifier"

# Per-stream drift-annotation overrides. Each entry may set:
#   ``ha``          horizontal alignment (default ``"center"``)
#   ``x_offset``    offset added to the drift x in data coords (default 0)
#   ``y_data``      override height in data y-coords (overrides axes-fraction)
_DRIFT_LABEL_OVERRIDES: Final[dict[str, dict]] = {
    "agrawal_drift_0to9": {"ha": "right", "x_offset": -40},
    "agrawal_drift_7to4": {"ha": "right", "x_offset": -30, "y_data": 0.30},
}


def _annotate_drift(
    ax: plt.Axes,
    x_pos: float,
    config: PlotConfig,
    *,
    override: dict | None = None,
) -> None:
    override = override or {}
    ha = override.get("ha", "center")
    x = x_pos + override.get("x_offset", 0)
    y_data = override.get("y_data")
    if y_data is not None:
        transform = ax.transData
        y = y_data
    else:
        transform = ax.get_xaxis_transform()
        y = _DRIFT_LABEL_Y
    ax.text(
        x,
        y,
        _DRIFT_LABEL,
        transform=transform,
        color=config.color_neutral,
        ha=ha,
        va="center",
        rotation=0,
        fontsize=10 * _FONT_SCALE,
        weight=500,
        bbox={"boxstyle": "round,pad=0.6", "fc": "white", "ec": config.color_gridline, "alpha": 0.95},
        zorder=10,
        clip_on=False,
    )


def _draw_drift_marker(
    ax: plt.Axes,
    data: dict,
    config: PlotConfig,
) -> None:
    """Shade the drift window if gradual; otherwise draw a vertical line at ``true_drift_t``.

    Annotates the marker with a text label rather than adding it to the legend.
    """
    override = _DRIFT_LABEL_OVERRIDES.get(data.get("stream", ""))
    drift_start = data.get("drift_start")
    drift_end = data.get("drift_end")
    if drift_start is not None and drift_end is not None:
        ax.axvspan(
            drift_start,
            drift_end,
            color=config.color_neutral,
            alpha=0.18,
            linewidth=0,
            zorder=0,
        )
        _annotate_drift(ax, (drift_start + drift_end) / 2, config, override=override)
        return
    drift_t = data.get("true_drift_t")
    if drift_t is None:
        return
    ax.axvline(
        drift_t,
        color=config.color_neutral,
        linestyle="--",
        lw=1.2,
        alpha=0.85,
        zorder=1,
    )
    _annotate_drift(ax, drift_t, config, override=override)


def _format_axes(ax: plt.Axes, config: PlotConfig) -> None:
    ax.grid(True, alpha=config.grid_alpha, linestyle=config.grid_linestyle, color=config.color_gridline)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _bottom_legend(fig: Figure, handles: list, config: PlotConfig) -> None:
    """Place a centred 4-column legend in the figure-level reserved bottom margin."""
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        edgecolor=config.color_gridline,
    )


def plot_decomposition(
    data: dict,
    config: PlotConfig | None = None,
    smooth_win: int = DEFAULT_SMOOTH_WIN,
    *,
    show_legend: bool = True,
    show_title: bool = True,
) -> Figure:
    """Plot total / aleatoric / epistemic medians with IQR shading.

    Args:
        data: Parsed JSON payload from ``run_stream.py``.
        config: Optional plotting configuration.
        smooth_win: Rolling-mean window applied per seed before aggregation.

    Returns:
        The matplotlib Figure.
    """
    cfg = config or PlotConfig()
    records = pd.DataFrame(data["records"])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=cfg.dpi)

    palette = {
        "TU": "#444444",
        "AU": cfg.color_negative,
        "EU": cfg.color_positive,
    }
    columns = (("total", "TU"), ("alea", "AU"), ("epi", "EU"))

    handles = []
    for col, label in columns:
        agg = _aggregate(records, col, smooth_win)
        color = palette[label]
        ax.fill_between(agg["t"], agg["lo"], agg["hi"], color=color, alpha=cfg.fill_alpha * 0.6, linewidth=0)
        (line,) = ax.plot(agg["t"], agg["med"], color=color, lw=cfg.line_width, label=label)
        handles.append(line)

    _draw_drift_marker(ax, data, cfg)
    _format_axes(ax, cfg)

    ax.set_xlabel("step $t$")
    ax.set_ylabel("uncertainty")
    ax.set_ylim(0, _uncertainty_ylim_top(records, ("total",), smooth_win))
    if show_title:
        ax.set_title(_FIGURE_TITLE)
    if show_legend:
        _bottom_legend(fig, handles, cfg)

    bottom = _BOTTOM_MARGIN if show_legend else _BOTTOM_MARGIN_NO_LEGEND
    top = 0.90 if show_title else 0.97
    fig.subplots_adjust(bottom=bottom, top=top, left=0.10, right=0.92)
    return fig


def plot_decomposition_accuracy(
    data: dict,
    config: PlotConfig | None = None,
    smooth_win: int = DEFAULT_SMOOTH_WIN,
    *,
    tu_color: str | None = None,
    acc_color: str | None = None,
    show_legend: bool = True,
    show_title: bool = True,
) -> Figure:
    """Plot total / aleatoric / epistemic on the left axis and rolling accuracy on the right.

    Combines the decomposition view with accuracy as a secondary signal on a
    twin axis. Useful when the relationship between uncertainty components
    and prediction accuracy is the focus of the figure.

    Args:
        data: Parsed JSON payload from ``run_stream.py``.
        config: Optional plotting configuration.
        smooth_win: Rolling-mean window applied per seed before aggregation.
        tu_color: Override for the TU (total uncertainty) line colour. Defaults
            to the dark-grey palette entry.
        acc_color: Override for the accuracy line colour and right-axis tint.
            Defaults to ``config.color_neutral``.

    Returns:
        The matplotlib Figure.
    """
    cfg = config or PlotConfig()
    records = pd.DataFrame(data["records"])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=cfg.dpi)

    palette = {
        "TU": tu_color or "#444444",
        "AU": cfg.color_negative,
        "EU": cfg.color_positive,
    }
    columns = (("total", "TU"), ("alea", "AU"), ("epi", "EU"))
    decomp_handles = []
    for col, label in columns:
        agg = _aggregate(records, col, smooth_win)
        color = palette[label]
        ax.fill_between(agg["t"], agg["lo"], agg["hi"], color=color, alpha=cfg.fill_alpha * 0.6, linewidth=0)
        (line,) = ax.plot(agg["t"], agg["med"], color=color, lw=cfg.line_width, label=label)
        decomp_handles.append(line)

    ax.set_xlabel("step $t$")
    ax.set_ylabel("uncertainty")
    ax.set_ylim(0, _uncertainty_ylim_top(records, ("total",), smooth_win))

    acc = _aggregate(records, "correct", int(smooth_win * _ACC_SMOOTH_FACTOR))
    acc_color = acc_color or _ACC_COLOR
    ax_right = ax.twinx()
    (acc_line,) = ax_right.plot(
        acc["t"],
        acc["med"],
        color=acc_color,
        lw=cfg.line_width,
        label="ACC",
        alpha=0.9,
    )
    ax_right.set_ylabel("rolling accuracy", color=_ACC_AXIS_COLOR)
    ax_right.tick_params(axis="y", labelcolor=_ACC_AXIS_COLOR)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.spines["top"].set_visible(False)

    _draw_drift_marker(ax, data, cfg)
    _format_axes(ax, cfg)

    handles = [*decomp_handles, acc_line]
    if show_legend:
        _bottom_legend(fig, handles, cfg)
    if show_title:
        ax.set_title(_FIGURE_TITLE)

    bottom = _BOTTOM_MARGIN if show_legend else _BOTTOM_MARGIN_NO_LEGEND
    top = 0.90 if show_title else 0.97
    fig.subplots_adjust(bottom=bottom, top=top, left=0.10, right=0.92)
    return fig


def plot_epi_accuracy(
    data: dict,
    config: PlotConfig | None = None,
    smooth_win: int = DEFAULT_SMOOTH_WIN,
    *,
    show_legend: bool = True,
    show_title: bool = True,
) -> Figure:
    """Plot epistemic uncertainty (left) and rolling accuracy (right) on twin axes.

    Args:
        data: Parsed JSON payload from ``run_stream.py``.
        config: Optional plotting configuration.
        smooth_win: Rolling-mean window applied per seed before aggregation.

    Returns:
        The matplotlib Figure.
    """
    cfg = config or PlotConfig()
    records = pd.DataFrame(data["records"])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=cfg.dpi)

    epi = _aggregate(records, "epi", smooth_win)
    acc = _aggregate(records, "correct", int(smooth_win * _ACC_SMOOTH_FACTOR))

    epi_color = cfg.color_positive
    acc_color = _ACC_COLOR

    ax.fill_between(epi["t"], epi["lo"], epi["hi"], color=epi_color, alpha=cfg.fill_alpha, linewidth=0)
    (epi_line,) = ax.plot(epi["t"], epi["med"], color=epi_color, lw=cfg.line_width, label="EU")
    ax.set_xlabel("step $t$")
    ax.set_ylabel("uncertainty", color=epi_color)
    ax.tick_params(axis="y", labelcolor=epi_color)
    ax.set_ylim(0, _uncertainty_ylim_top(records, ("epi",), smooth_win))

    ax_right = ax.twinx()
    (acc_line,) = ax_right.plot(acc["t"], acc["med"], color=acc_color, lw=cfg.line_width, label="ACC")
    ax_right.set_ylabel("rolling accuracy", color=_ACC_AXIS_COLOR)
    ax_right.tick_params(axis="y", labelcolor=_ACC_AXIS_COLOR)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.spines["top"].set_visible(False)

    _draw_drift_marker(ax, data, cfg)
    _format_axes(ax, cfg)

    handles = [epi_line, acc_line]
    if show_legend:
        _bottom_legend(fig, handles, cfg)
    if show_title:
        ax.set_title(_FIGURE_TITLE)

    bottom = _BOTTOM_MARGIN if show_legend else _BOTTOM_MARGIN_NO_LEGEND
    top = 0.90 if show_title else 0.97
    fig.subplots_adjust(bottom=bottom, top=top, left=0.10, right=0.92)
    return fig


# ----- legend-only figures (for placing the legend separately in LaTeX) -----


def _legend_only_figure(items: list[tuple[str, str]], config: PlotConfig | None = None) -> Figure:
    """Build a slim figure containing only a horizontal legend."""
    cfg = config or PlotConfig()
    fig = plt.figure(figsize=_LEGEND_FIGSIZE, dpi=cfg.dpi)
    handles = [
        mpl.lines.Line2D([0], [0], color=color, lw=cfg.line_width, label=label)
        for label, color in items
    ]
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(items),
        frameon=False,
        edgecolor=cfg.color_gridline,
    )
    return fig


def legend_decomposition(config: PlotConfig | None = None) -> Figure:
    """Standalone legend matching :func:`plot_decomposition` (TU/AU/EU)."""
    cfg = config or PlotConfig()
    return _legend_only_figure(
        [("TU", "#444444"), ("AU", cfg.color_negative), ("EU", cfg.color_positive)],
        cfg,
    )


def legend_combined(config: PlotConfig | None = None) -> Figure:
    """Standalone legend matching :func:`plot_decomposition_accuracy` (TU/AU/EU/ACC)."""
    cfg = config or PlotConfig()
    return _legend_only_figure(
        [
            ("TU", "#444444"),
            ("AU", cfg.color_negative),
            ("EU", cfg.color_positive),
            ("ACC", _ACC_COLOR),
        ],
        cfg,
    )


def legend_combined_alt(config: PlotConfig | None = None) -> Figure:
    """Standalone legend matching the alt combined plot (TU=green, rest unchanged)."""
    cfg = config or PlotConfig()
    return _legend_only_figure(
        [
            ("TU", "#2ecc71"),
            ("AU", cfg.color_negative),
            ("EU", cfg.color_positive),
            ("ACC", _ACC_COLOR),
        ],
        cfg,
    )


def legend_accuracy(config: PlotConfig | None = None) -> Figure:
    """Standalone legend matching :func:`plot_epi_accuracy` (EU/ACC)."""
    cfg = config or PlotConfig()
    return _legend_only_figure(
        [("EU", cfg.color_positive), ("ACC", _ACC_COLOR)],
        cfg,
    )


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {path}")


def _process_one(json_path: Path, out_dir: Path, smooth_win: int) -> None:
    with json_path.open() as fh:
        data = json.load(fh)
    stream = data["stream"]

    # decomposition (TU / AU / EU)
    _save(plot_decomposition(data, smooth_win=smooth_win), out_dir / f"{stream}_decomposition.pdf")
    _save(
        plot_decomposition(data, smooth_win=smooth_win, show_legend=False),
        out_dir / f"{stream}_decomposition_nolegend.pdf",
    )

    # epi + accuracy
    _save(plot_epi_accuracy(data, smooth_win=smooth_win), out_dir / f"{stream}_accuracy.pdf")
    _save(
        plot_epi_accuracy(data, smooth_win=smooth_win, show_legend=False),
        out_dir / f"{stream}_accuracy_nolegend.pdf",
    )

    # combined (TU / AU / EU + ACC)
    _save(
        plot_decomposition_accuracy(data, smooth_win=smooth_win),
        out_dir / f"{stream}_combined.pdf",
    )
    _save(
        plot_decomposition_accuracy(data, smooth_win=smooth_win, show_legend=False),
        out_dir / f"{stream}_combined_nolegend.pdf",
    )

    # combined alt (TU = green)
    _save(
        plot_decomposition_accuracy(data, smooth_win=smooth_win, tu_color="#2ecc71"),
        out_dir / f"{stream}_combined_alt.pdf",
    )
    _save(
        plot_decomposition_accuracy(
            data, smooth_win=smooth_win, tu_color="#2ecc71", show_legend=False
        ),
        out_dir / f"{stream}_combined_alt_nolegend.pdf",
    )


def _save_standalone_legends(out_dir: Path) -> None:
    """Write four legend-only PDFs (one per plot type) into ``out_dir``."""
    _save(legend_decomposition(), out_dir / "legend_decomposition.pdf")
    _save(legend_accuracy(), out_dir / "legend_accuracy.pdf")
    _save(legend_combined(), out_dir / "legend_combined.pdf")
    _save(legend_combined_alt(), out_dir / "legend_combined_alt.pdf")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Path to a single <stream>.json file. Mutually exclusive with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every <stream>.json under --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Directory of JSON inputs (and PDF outputs unless --out-dir is set).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write PDFs. Defaults to --results-dir.",
    )
    parser.add_argument(
        "--smooth-win",
        type=int,
        default=DEFAULT_SMOOTH_WIN,
        help="Rolling-mean window applied per seed before aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.all == bool(args.json_path):
        msg = "Pass exactly one of --json-path or --all."
        raise SystemExit(msg)

    out_dir: Path = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        json_paths = sorted(args.results_dir.glob("*.json"))
        if not json_paths:
            msg = f"no *.json files in {args.results_dir}"
            raise SystemExit(msg)
    else:
        if not args.json_path.exists():
            msg = f"json not found: {args.json_path}"
            raise SystemExit(msg)
        json_paths = [args.json_path]

    for path in json_paths:
        _process_one(path, out_dir, args.smooth_win)

    _save_standalone_legends(out_dir)


if __name__ == "__main__":
    main()
