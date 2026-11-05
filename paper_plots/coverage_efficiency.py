"""Coverage vs. efficiency of credal methods on first-order CIFAR-10 data.

Each method is plotted in the coverage notion of its own credal set: interval
coverage for EffCre, CreRL, CreWra, and CreNet, and convex hull coverage for CreEns
and CreBNN. Efficiency is ``1 - mean(upper - lower)``, so higher is better on both
axes. Methods with an alpha sweep are lines; the others are single points.

Reads only the local caches. Run ``python -m paper_plots.import_csv`` first.
"""

from __future__ import annotations

from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np

from paper_plots import cache, sweep_cache
from paper_plots.fetch_first_order import ALLOWED_DATASETS
from paper_plots.metadata import dataset_display, method_label
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import PlotConfig
from paper_plots.targets import ENTITY, TARGETS, Target

FIGSIZE: tuple[float, float] = (4.5, 3.5)
# The runs train on CIFAR-10, but coverage is measured against the first-order CIFAR-10-H labels.
TITLE_DATASETS: dict[str, str] = {"cifar10": "CIFAR-10-H"}
# This plot uses regular text, one step heavier than the light default of the other figures.
TEXT_WEIGHT = 400
XLIM: tuple[float, float] = (-0.03, 1.03)
YLIM: tuple[float, float] = (-0.03, 1.09)
TICKS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Both metrics end at 1. The limits go further only to leave room for the alpha labels.
AXIS_END = 1.0

# The sweeps, then the single-point methods; colors follow this order.
LINES: tuple[str, ...] = (
    "efficient_credal_prediction",
    "credal_relative_likelihood",
    "credal_ensembling",
)
POINTS: tuple[str, ...] = ("credal_wrapper", "credal_bnn", "credal_net")
COVERAGE_KEYS: dict[str, str] = {
    "credal_ensembling": "convex_hull_coverage",
    "credal_bnn": "convex_hull_coverage",
}
# At alpha=0 the set is the whole simplex: coverage 1 and efficiency 0 by definition, not a run.
ALPHA_ZERO_POINT: tuple[float, float] = (1.0, 0.0)
ALPHA_ZERO_POINT_METHODS: frozenset[str] = frozenset({"efficient_credal_prediction", "credal_relative_likelihood"})

_BELOW = ((0.0, -9.0), "center")
_BELOW_LEFT = ((-2.0, -7.0), "right")
# Alphas to print per sweep: alpha -> (offset in points, horizontal alignment).
# EffCre and CreRL also share a pair of alpha=0 labels, drawn separately.
ALPHA_LABELS: dict[str, dict[float, tuple[tuple[float, float], str]]] = {
    "efficient_credal_prediction": {
        # Above its point, unlike the others: the CreRL line converges here and cuts through the usual spot.
        0.2: ((-3.5, 5.0), "right"),
        0.4: _BELOW_LEFT,
        0.6: _BELOW_LEFT,
        0.8: _BELOW_LEFT,
        0.9: _BELOW_LEFT,
        0.95: _BELOW_LEFT,
        1.0: _BELOW,
    },
    # Stepped down from .8 to .2, or the labels run into their neighbors.
    "credal_relative_likelihood": {
        0.2: ((-1.0, -12.0), "right"),
        0.4: ((0.0, -12.0), "right"),
        0.6: ((0.5, -10.5), "right"),
        0.8: ((1.0, -8.5), "right"),
        0.9: _BELOW_LEFT,
        0.95: _BELOW_LEFT,
        1.0: ((-6.0, 3.5), "right"),
    },
    # The two share a height: alpha=0 sits 4 points lower in the data, so its offset is 4 points larger.
    "credal_ensembling": {
        0.0: ((5.5, 6.5), "left"),
        1.0: ((-4.5, 2.5), "right"),
    },
}


def _alpha_label(alpha: float) -> str:
    return rf"$\alpha$={alpha:g}"


def _mean(records: list[dict], key: str) -> float | None:
    values = [float(r["summary"][key]) for r in records if r.get("summary", {}).get(key) is not None]
    return float(np.mean(values)) if values else None


def _point(records: list[dict], method: str) -> dict[str, float] | None:
    coverage = _mean(records, COVERAGE_KEYS.get(method, "coverage"))
    efficiency = _mean(records, "efficiency")
    if coverage is None or efficiency is None:
        return None
    return {"cov": coverage, "eff": efficiency}


def _sweep(target: Target, method: str) -> list[dict[str, float]]:
    """Return one point per alpha, ascending, starting at the alpha=0 point where it applies."""
    args = (ENTITY, target.project, target.dataset, target.base_model, method)
    points = []
    if method in ALPHA_ZERO_POINT_METHODS:
        points.append({"alpha": 0.0, "cov": ALPHA_ZERO_POINT[0], "eff": ALPHA_ZERO_POINT[1]})
    for alpha in sweep_cache.list_alphas(*args):
        point = _point(sweep_cache.load_records(*args, alpha), method)
        if point is not None:
            points.append({"alpha": alpha, **point})
    return points


def _render_one(target: Target) -> None:
    cfg = PlotConfig()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for index, method in enumerate(LINES):
        color = cfg.color(index)
        points = _sweep(target, method)
        if not points:
            warnings.warn(f"no sweep cached for {method} on {target.dataset}", stacklevel=2)
            continue
        ax.plot(
            [p["cov"] for p in points],
            [p["eff"] for p in points],
            color=color,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=cfg.line_width,
            zorder=2,
            label=method_label(method),
        )
        for point in points:
            placement = ALPHA_LABELS[method].get(point["alpha"])
            if placement is not None:
                offset, ha = placement
                ax.annotate(
                    _alpha_label(point["alpha"]),
                    (point["cov"], point["eff"]),
                    xytext=offset,
                    textcoords="offset points",
                    ha=ha,
                    va="center",
                    fontsize=7,
                    fontweight=TEXT_WEIGHT,
                    color=color,
                    zorder=5,
                )

    # EffCre and CreRL end in the same alpha=0 point, so their labels share one spot: "a=0, a=0".
    # Baseline alignment keeps them level; centering would lift the one with the comma's descender.
    for text, right_edge, color in (
        (_alpha_label(0.0), -6.5, cfg.color(0)),
        (f"{_alpha_label(0.0)},", -19.5, cfg.color(1)),
    ):
        ax.annotate(
            text,
            ALPHA_ZERO_POINT,
            xytext=(right_edge, -2.5),
            textcoords="offset points",
            ha="right",
            va="baseline",
            fontsize=7,
            fontweight=TEXT_WEIGHT,
            color=color,
            zorder=5,
        )

    for index, method in enumerate(POINTS, start=len(LINES)):
        records = cache.load_records(ENTITY, target.project, target.dataset, target.base_model, method)
        point = _point(records, method)
        if point is None:
            warnings.warn(f"no coverage or efficiency cached for {method} on {target.dataset}", stacklevel=2)
            continue
        ax.scatter(
            point["cov"],
            point["eff"],
            s=60,
            color=cfg.color(index),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=method_label(method),
        )

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Coverage $\mathbf{\rightarrow}$")
    ax.set_ylabel(r"Efficiency $\mathbf{\rightarrow}$")
    ax.set_title(f"Coverage vs. efficiency on {TITLE_DATASETS.get(target.dataset, dataset_display(target.dataset))}")
    # Drawn by hand because a regular grid runs on to the axis limits, past the values both metrics can reach.
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    grid_style: dict[str, Any] = {
        "colors": cfg.color_gridline,
        "linestyles": cfg.grid_linestyle,
        "alpha": cfg.grid_alpha,
        "linewidths": 0.8,
        "zorder": 0.5,
    }
    ax.hlines(TICKS, XLIM[0], AXIS_END, **grid_style)
    ax.vlines(TICKS, YLIM[0], AXIS_END, **grid_style)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(XLIM[0], AXIS_END)
    ax.spines["left"].set_bounds(YLIM[0], AXIS_END)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight(TEXT_WEIGHT)
    ax.legend(
        loc="lower left",
        frameon=False,
        labelspacing=0.48,
        handlelength=1.6,
        prop={"size": 8, "weight": TEXT_WEIGHT},
    )
    # This figure prints at about 40% of its size, so margins are kept small to let the content print larger.
    fig.tight_layout(pad=0.4)

    out_dir = PAPER_ARTIFACTS / f"{target.dataset}_methods"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"coverage_efficiency_{target.dataset}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    for target in TARGETS:
        if target.dataset in ALLOWED_DATASETS:
            _render_one(target)


if __name__ == "__main__":
    main()
