"""Active-learning ranked bar plot, one PDF per (dataset, notion).

Each panel shows the 19 UQ methods at ``strategy=uncertainty`` plus the loss
/ calibration variants of ``base`` (also at strategy=margin), all sorted
together by mean NAUC. Bars share a single colour -- the variant rows are
no longer distinguished by hue.

Two markers per UQ bar pin the **same method's** ``random`` and ``margin``
NAUC -- this grounds how much of the gain comes from the UQ signal versus the
model architecture. Variant bars don't carry markers (their bar value already
sits at ``strategy=margin``).

Two horizontal dashed lines show the pure-strategy baselines (``base +
random`` and ``base + margin`` with cross_entropy / cal=none). The full
baseline set is defined by :data:`paper_plots.al_aggregate.BASE_VARIANTS`;
each entry's ``render`` field decides whether it's drawn as a bar or as a
horizontal reference line.

Reads only the merged AL cache. Run::

    uv run python -m paper_plots.fetch_al
    uv run python -m paper_plots.analyze_al
    uv run python -m paper_plots.bar_al
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from paper_plots.al_aggregate import BASE_VARIANTS, base_variant_levels, uq_ranking
from paper_plots.metadata import dataset_display, method_label
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import PlotConfig

DATASETS: tuple[str, ...] = ("openml_6", "openml_156")
NOTIONS: tuple[str, ...] = ("epistemic", "aleatoric", "total")
NOTION_LABEL = {"epistemic": "EU", "aleatoric": "AU", "total": "TU"}

OUT_DIR = PAPER_ARTIFACTS / "al_methods"
# Smaller figure so two bar plots sit comfortably side by side.
FIGSIZE: tuple[float, float] = (8.0, 4.0)

# Single bar colour for both UQ methods and the loss/calibration variant bars.
_BAR_COLOR = "#1e88e5"  # blue, matches PlotConfig.color(0)

# Reference-line styling for the two pure-strategy baselines drawn as
# horizontal lines instead of bars. ``base + margin`` shares the red of the
# margin marker so the eye links the two; ``base + random`` is neutral grey.
_LINE_STYLE: dict[str, dict[str, object]] = {
    "base_random": {"linestyle": (0, (4, 3)), "color": "#7f8c8d"},
    "base_margin": {"linestyle": (0, (5, 1, 1, 1)), "color": "#ff0d57"},
}

# Per-method ablation marker styling.
_RANDOM_MARKER_COLOR = "#999999"  # light grey diamond
_MARGIN_MARKER_COLOR = "#ff0d57"  # red square (matches base_margin reference line)


def _ylim(combined: list[dict], variants: dict[str, dict]) -> tuple[float, float]:
    values: list[float] = [e["mean"] for e in combined]
    values.extend(e[k] for e in combined for k in ("random_mean", "margin_mean") if e.get(k) is not None)
    values.extend(v["mean"] for v in variants.values() if v["mean"] is not None)
    if not values:
        return 0.0, 1.0
    lo = max(0.0, min(values) - 0.05)
    hi = min(1.0, max(values) + 0.05)
    if hi - lo < 0.1:
        hi = min(1.0, lo + 0.1)
    return lo, hi


def _render_panel(dataset: str, notion: str) -> None:  # noqa: PLR0915
    cfg = PlotConfig()
    ranking = uq_ranking(dataset, notion)
    if not ranking:
        warnings.warn(f"no UQ rankings for {dataset}/{notion}", stacklevel=2)
        return
    variants = base_variant_levels(dataset)

    # Build the combined leaderboard: UQ bars + variant bars in one sorted list.
    combined: list[dict] = [{**e, "kind": "uq"} for e in ranking]
    for variant in BASE_VARIANTS:
        if variant.get("render") != "bar":
            continue
        info = variants.get(variant["key"])
        if info is None or info["mean"] is None:
            continue
        # x-tick label for bar variants: just the pretty loss/cal name. The
        # acquisition strategy is implicit (margin) and would crowd the axis.
        bar_label = method_label(variant["key"])
        combined.append(
            {
                "method": variant["key"],
                "label": bar_label,
                "mean": info["mean"],
                "std": info["std"],
                "n": info["n"],
                "random_mean": None,
                "margin_mean": None,
                "kind": "variant",
            },
        )
    combined.sort(key=lambda r: r["mean"], reverse=True)

    labels = [e["label"] for e in combined]
    means = [e["mean"] for e in combined]
    stds = [e["std"] for e in combined]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bar_width = 0.78
    ax.bar(
        x,
        means,
        width=bar_width,
        yerr=stds,
        capsize=3,
        color=_BAR_COLOR,
        edgecolor="none",
        error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capthick": 1.0},
        zorder=3,
    )

    # Per-method ablation markers: random + margin NAUC for the same UQ method.
    # Variant bars skip these (their bar already represents the margin point).
    rand_x = [xi for xi, e in zip(x, combined, strict=True) if e["kind"] == "uq" and e["random_mean"] is not None]
    rand_y = [e["random_mean"] for e in combined if e["kind"] == "uq" and e["random_mean"] is not None]
    if rand_x:
        ax.scatter(
            rand_x,
            rand_y,
            marker="D",
            color=_RANDOM_MARKER_COLOR,
            s=28,
            zorder=5,
            label="this method + random",
        )
    marg_x = [xi for xi, e in zip(x, combined, strict=True) if e["kind"] == "uq" and e["margin_mean"] is not None]
    marg_y = [e["margin_mean"] for e in combined if e["kind"] == "uq" and e["margin_mean"] is not None]
    if marg_x:
        ax.scatter(
            marg_x,
            marg_y,
            marker="s",
            color=_MARGIN_MARKER_COLOR,
            s=24,
            zorder=5,
            label="this method + margin",
        )

    # Reference lines: only BASE_VARIANTS entries marked render="line".
    for variant in BASE_VARIANTS:
        if variant.get("render") != "line":
            continue
        info = variants.get(variant["key"])
        if info is None or info["mean"] is None:
            continue
        style = _LINE_STYLE[variant["key"]]
        ax.axhline(
            info["mean"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=1.2,
            zorder=2,
            label=info["label"],
        )

    inter_bar_gap = 1.0 - bar_width
    ax.set_xlim(-bar_width / 2 - inter_bar_gap, len(labels) - 1 + bar_width / 2 + inter_bar_gap)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("NAUC")
    ax.set_ylim(*_ylim(combined, variants))
    ax.set_title(
        f"Active learning on {dataset_display(dataset)}  $\\mathrm{{({NOTION_LABEL[notion]})}}$",
    )
    ax.yaxis.grid(visible=True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha, color=cfg.color_gridline)
    ax.set_axisbelow(True)

    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"bar_al_{dataset}_{notion}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    for dataset in DATASETS:
        for notion in NOTIONS:
            _render_panel(dataset, notion)


if __name__ == "__main__":
    main()
