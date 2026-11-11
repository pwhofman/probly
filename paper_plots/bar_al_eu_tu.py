"""Combined TU + EU AL bar plot, one PDF per dataset.

For each UQ method we draw two bars at the same x and the same width:

- the **larger** of (TU, EU) goes underneath (lower zorder),
- the **smaller** sits on top (higher zorder),

so both heights are always visible -- the larger one peeks above the smaller
one regardless of which is bigger. The TU bar gets the regular colour
(blue / green for variants) and the EU bar uses a lighter shade of the same
colour. Variant bars (loss / calibration) are notion-agnostic and only have
the TU colour.

Methods are sorted by the taller of their two bars, so the panel steps down from
left to right whichever notion a method does best with.

Notion-agnostic baselines (loss / calibration variants of ``base``) only have
one NAUC value -- no EU bar is drawn behind them. The two pure-strategy
baselines (``base + random`` / ``base + margin``) are still rendered as the
two horizontal dashed reference lines, identical to :mod:`paper_plots.bar_al`.

A method without a TU run fills the TU bar from its AU or EU run.

Reads only the merged AL cache. Run::

    uv run python -m paper_plots.bar_al_eu_tu
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from paper_plots.al_aggregate import BASE_VARIANTS, base_variant_levels, uq_ranking
from paper_plots.metadata import dataset_display, method_label
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import LEGEND_ROW_PAD, PlotConfig, draw_legend_row, draw_whiskers, fit_ylim, round_bar_tops

if TYPE_CHECKING:
    from matplotlib.patches import Rectangle

DATASETS: tuple[str, ...] = ("openml_6", "openml_156", "cifar10")
OUT_DIR = PAPER_ARTIFACTS / "al_methods"
FIGSIZE: tuple[float, float] = (8.0, 4.0)

# Single bar colour shared by UQ methods and loss/calibration variants
# (mirrors bar_al.py).
_BAR_COLOR = "#1e88e5"  # blue, matches PlotConfig.color(0)

# Reference-line styling for the two pure-strategy baselines (mirrors bar_al).
_LINE_STYLE: dict[str, dict[str, object]] = {
    "base_random": {"linestyle": "-", "color": "#7f8c8d"},
    "base_margin": {"linestyle": (0, (5, 1, 1, 1)), "color": "#ff0d57"},
}

# Per-method ablation marker styling (mirrors bar_al).
_RANDOM_MARKER_COLOR = "#999999"
_MARGIN_MARKER_COLOR = "#ff0d57"


def _lighter(color: str, mix: float = 0.55) -> tuple[float, float, float]:
    """Mix ``color`` with white. ``mix=0`` keeps it, ``mix=1`` returns white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * mix, g + (1 - g) * mix, b + (1 - b) * mix)


def _drawn_heights(combined: list[dict], variants: dict[str, dict]) -> list[float]:
    """Heights of all bars, whisker ends, markers and reference lines in a panel."""
    heights: list[float] = []
    for entry in combined:
        heights += [entry["tu_mean"] - entry["tu_std"], entry["tu_mean"] + entry["tu_std"]]
        heights += [entry[key] for key in ("eu_mean", "random_mean", "margin_mean") if entry.get(key) is not None]
    for variant in BASE_VARIANTS:
        info = variants.get(variant["key"])
        if variant.get("render") == "line" and info is not None and info["mean"] is not None:
            heights.append(info["mean"])
    return heights


def _render_panel(dataset: str) -> None:  # noqa: C901, PLR0912, PLR0915
    cfg = PlotConfig()
    tu_ranking = uq_ranking(dataset, "total", fallback=True)
    if not tu_ranking:
        warnings.warn(f"no TU rankings for {dataset}", stacklevel=2)
        return
    eu_ranking = uq_ranking(dataset, "epistemic")
    eu_by_method: dict[str, dict] = {entry["method"]: entry for entry in eu_ranking}
    variants = base_variant_levels(dataset)

    combined: list[dict] = []
    for entry in tu_ranking:
        eu_entry = eu_by_method.get(entry["method"])
        combined.append(
            {
                "method": entry["method"],
                "label": entry["label"],
                "tu_mean": entry["mean"],
                "tu_std": entry["std"],
                "eu_mean": eu_entry["mean"] if eu_entry else None,
                "random_mean": entry.get("random_mean"),
                "margin_mean": entry.get("margin_mean"),
                "kind": "uq",
            },
        )

    for variant in BASE_VARIANTS:
        if variant.get("render") != "bar":
            continue
        info = variants.get(variant["key"])
        if info is None or info["mean"] is None:
            continue
        combined.append(
            {
                "method": variant["key"],
                "label": method_label(variant["key"]),
                "tu_mean": info["mean"],
                "tu_std": info["std"],
                "eu_mean": None,  # variants are notion-agnostic; no EU overlay
                "random_mean": None,
                "margin_mean": None,
                "kind": "variant",
            },
        )
    combined.sort(key=lambda r: max(r["tu_mean"], r["eu_mean"] or 0.0), reverse=True)

    labels = [e["label"] for e in combined]
    tu_means = [e["tu_mean"] for e in combined]
    tu_stds = [e["tu_std"] for e in combined]
    fg_colors = [_BAR_COLOR for _ in combined]
    bg_colors = [_lighter(_BAR_COLOR) for _ in combined]
    x = np.arange(len(labels))
    bar_width = 0.78

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Per-bar TU/EU pair: bigger bar behind, smaller bar on top, both at the
    # same x and the same width so the bottoms align pixel-perfectly.
    back_x: list[int] = []
    back_h: list[float] = []
    back_c: list[tuple[float, float, float] | str] = []
    front_x: list[int] = []
    front_h: list[float] = []
    front_c: list[tuple[float, float, float] | str] = []
    for xi, entry, fg, bg in zip(x, combined, fg_colors, bg_colors, strict=True):
        tu = entry["tu_mean"]
        eu = entry.get("eu_mean")
        if eu is None:
            # Variant: only TU bar (no EU axis). Treat as foreground only.
            front_x.append(int(xi))
            front_h.append(tu)
            front_c.append(fg)
            continue
        if np.isclose(tu, eu):
            # Same run behind both bars: one shape is enough, and a twin behind it
            # would show as a dark fringe around the rounded corners.
            front_x.append(int(xi))
            front_h.append(eu)
            front_c.append(bg)
            continue
        if tu >= eu:
            # TU bigger -> behind, EU smaller -> front.
            back_x.append(int(xi))
            back_h.append(tu)
            back_c.append(fg)
            front_x.append(int(xi))
            front_h.append(eu)
            front_c.append(bg)
        else:
            # EU bigger -> behind, TU smaller -> front.
            back_x.append(int(xi))
            back_h.append(eu)
            back_c.append(bg)
            front_x.append(int(xi))
            front_h.append(tu)
            front_c.append(fg)

    bars: list[Rectangle] = []
    if back_x:
        bars.extend(ax.bar(back_x, back_h, width=bar_width, color=back_c, edgecolor="none", linewidth=0, zorder=2))
    if front_x:
        # Stay below the axis line (zorder 2.5), or the bars paint over its upper half.
        bars.extend(ax.bar(front_x, front_h, width=bar_width, color=front_c, edgecolor="none", linewidth=0, zorder=2.2))

    # Whiskers at TU height, drawn separately so they don't depend on which bar is in front.
    draw_whiskers(ax, x, tu_means, tu_stds, linewidth=1.0)

    # Per-method ablation markers: random + margin NAUC for the same UQ method.
    rand_x = [xi for xi, e in zip(x, combined, strict=True) if e["kind"] == "uq" and e.get("random_mean") is not None]
    rand_y = [e["random_mean"] for e in combined if e["kind"] == "uq" and e.get("random_mean") is not None]
    if rand_x:
        ax.scatter(
            rand_x,
            rand_y,
            marker="D",
            color=_RANDOM_MARKER_COLOR,
            s=28,
            zorder=6,
            label="method + random",
        )
    marg_x = [xi for xi, e in zip(x, combined, strict=True) if e["kind"] == "uq" and e.get("margin_mean") is not None]
    marg_y = [e["margin_mean"] for e in combined if e["kind"] == "uq" and e.get("margin_mean") is not None]
    if marg_x:
        ax.scatter(
            marg_x,
            marg_y,
            marker="s",
            color=_MARGIN_MARKER_COLOR,
            s=24,
            zorder=6,
            label="method + margin",
        )

    # Reference lines: BASE_VARIANTS entries marked render="line".
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
            zorder=1,
            label=info["label"],
        )

    inter_bar_gap = 1.0 - bar_width
    ax.set_xlim(-bar_width / 2 - inter_bar_gap, len(labels) - 1 + bar_width / 2 + inter_bar_gap)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("NAUC")
    fit_ylim(ax, _drawn_heights(combined, variants))
    ax.set_title(
        f"Active learning on {dataset_display(dataset)}  $\\mathrm{{(TU\\ and\\ EU)}}$",
        pad=LEGEND_ROW_PAD,
    )
    ax.yaxis.grid(visible=True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha, color=cfg.color_gridline)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Compose legend: TU + EU swatches (so the two-colour overlay is readable)
    # then the auto handles for markers + reference lines.
    line_handles, _ = ax.get_legend_handles_labels()
    # Per-method markers before base lines, and margin before random within each pair.
    line_handles.sort(key=lambda h: (h.get_label().startswith("base"), "random" in h.get_label()))
    draw_legend_row(
        ax,
        [
            Patch(facecolor=_BAR_COLOR, label="TU"),
            Patch(facecolor=_lighter(_BAR_COLOR), label="EU"),
            *line_handles,
        ],
        columnspacing=0.9,
    )
    fig.tight_layout()
    round_bar_tops(ax, bars)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"bar_al_eu_tu_{dataset}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    for dataset in DATASETS:
        _render_panel(dataset)


if __name__ == "__main__":
    main()
