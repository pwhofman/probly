"""Top-K active-learning learning curves per (dataset, notion).

For each ``(dataset, notion)`` panel: the top-K UQ methods (ranked by mean
NAUC, computed live from the merged AL cache) are overlaid as colored
``test_accuracy`` vs ``labeled_size`` curves with a ``+/- std`` shaded band
across seeds. Two dashed grey reference curves are drawn underneath: the
pure-strategy baselines ``base + random`` and ``base + margin`` from
:data:`paper_plots.al_aggregate.BASE_VARIANTS`. Loss / calibration variants
are intentionally not plotted here -- they live in the AL bar plots only.

Reads only the merged AL cache. Run::

    uv run python -m paper_plots.fetch_al
    uv run python -m paper_plots.analyze_al
    uv run python -m paper_plots.al_curves
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from paper_plots import al_cache
from paper_plots.al_aggregate import (
    BASE_VARIANTS,
    DEFAULT_CAL,
    DEFAULT_LOSS,
    SINK,
    uq_ranking,
)
from paper_plots.metadata import BASE_VARIANT_KEYS, dataset_display, method_label
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import PlotConfig
from paper_plots.targets import ENTITY

# --- edit me -----------------------------------------------------------------
DATASETS: tuple[str, ...] = ("openml_6", "openml_156", "cifar10")
NOTIONS: tuple[str, ...] = ("epistemic", "aleatoric", "total")
NOTION_LABEL = {"epistemic": "EU", "aleatoric": "AU", "total": "TU"}
# Panels to leave out: on CIFAR-10 only DDU, PostNet and HetNet ran with AU acquisition.
SKIPPED_PANELS: frozenset[tuple[str, str]] = frozenset({("cifar10", "aleatoric")})
TOP_K: int | None = 10
# Optional bottom-K selection (worst NAUC). When set, the union of top + bottom
# is plotted (top first, then any bottom methods not already in top).
BOTTOM_K: int | None = None
OUT_DIR = PAPER_ARTIFACTS / "al_methods"
SEEDS: list[int] | None = None
# Shared single-panel default with the other "squarish" plots (sp_arc,
# coverage_efficiency, ood_hist).
FIGSIZE: tuple[float, float] = (4.5, 3.6)
# Legend layout. Override (e.g. via :mod:`paper_plots.al_curves_one_col`) to
# regenerate the panels with a different legend shape.
LEGEND_NCOL: int = 2
# Optional suffix appended to the output filename stem (before ``_<dataset>``).
FILENAME_SUFFIX: str = ""
# -----------------------------------------------------------------------------


def _curve_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Stack history rows across seeds into ``(labeled_size, mean_acc, std_acc)``.

    Seeds with non-modal history length are dropped (silently) so the stack
    is rectangular.
    """
    histories = [r.get("history", []) for r in records if r.get("history")]
    if not histories:
        return None
    expected_len = len(histories[0])
    aligned = [h for h in histories if len(h) == expected_len]
    if not aligned:
        return None
    x = np.asarray([row["labeled_size"] for row in aligned[0]], dtype=float)
    accuracy = np.asarray([[row["test_accuracy"] for row in h] for h in aligned], dtype=float)
    mean = accuracy.mean(axis=0)
    std = accuracy.std(axis=0) if accuracy.shape[0] > 1 else np.zeros_like(mean)
    return x, mean, std


def _mean_nauc(records: list[dict]) -> float | None:
    """Mean ``summary.nauc`` across records, or ``None`` when no parseable NAUC."""
    values: list[float] = []
    for r in records:
        v = r.get("summary", {}).get("nauc")
        if v is None:
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else None


def _baseline_curves(dataset: str) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Pure-strategy baseline curves (``base + random`` and ``base + margin``).

    Loss / calibration variants are deliberately skipped here -- they live in
    the AL bar plots only, not in the curves. Returns ``(label, x, mean, std)``
    per curve, with the legend label suffixed by mean NAUC. Empty curves are
    dropped.
    """
    out: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for variant in BASE_VARIANTS:
        if variant["key"] in BASE_VARIANT_KEYS:
            continue  # skip loss/calibration variants
        records = al_cache.load_merged(
            ENTITY,
            SINK,
            dataset,
            "base",
            variant["strategy"],
            supervised_loss=variant["loss"],
            calibration=variant["calibration"],
            seeds=SEEDS,
        )
        arrays = _curve_arrays(records)
        if arrays is None:
            continue
        x, mean, std = arrays
        nauc = _mean_nauc(records)
        label = f"{variant['label']} (NAUC={nauc:.3f})" if nauc is not None else variant["label"]
        out.append((label, x, mean, std))
    return out


def _selected_uq_curves(
    dataset: str,
    notion: str,
    *,
    top_k: int | None,
    bottom_k: int | None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Curves for the top-K plus bottom-K UQ methods by NAUC.

    When both are ``None`` the full ranking is used. Otherwise the union of
    the two slices is plotted (top first, then any bottom methods not already
    in top), order preserved by NAUC.
    """
    ranking = uq_ranking(dataset, notion)
    if top_k is None and bottom_k is None:
        selected = ranking
    else:
        top = ranking[:top_k] if top_k else []
        bot = ranking[-bottom_k:] if bottom_k else []
        seen: set[str] = set()
        selected = []
        for entry in (*top, *bot):
            if entry["method"] in seen:
                continue
            seen.add(entry["method"])
            selected.append(entry)

    out: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for entry in selected:
        records = al_cache.load_merged(
            ENTITY,
            SINK,
            dataset,
            entry["method"],
            "uncertainty",
            notion=notion,
            supervised_loss=DEFAULT_LOSS,
            calibration=DEFAULT_CAL,
            seeds=SEEDS,
        )
        arrays = _curve_arrays(records)
        if arrays is None:
            continue
        x, mean, std = arrays
        label = f"{method_label(entry['method'])} (NAUC={entry['mean']:.3f})"
        out.append((label, x, mean, std))
    return out


def _render_panel(dataset: str, notion: str) -> None:
    cfg = PlotConfig()
    uq_curves = _selected_uq_curves(dataset, notion, top_k=TOP_K, bottom_k=BOTTOM_K)
    baseline_curves = _baseline_curves(dataset)
    if not uq_curves and not baseline_curves:
        warnings.warn(f"no AL curves for {dataset}/{notion}", stacklevel=2)
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Pure-strategy baselines (random / margin) drawn as grey curves
    # under the UQ method curves. Each gets its own legend entry.
    grey = plt.get_cmap("Greys")
    n_baselines = max(1, len(baseline_curves))
    for idx, (label, x, mean, std) in enumerate(baseline_curves):
        color = grey(0.45 + 0.30 * idx / max(1, n_baselines - 1))
        linestyle = "-" if label.startswith("base + random") else "--"
        ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.3, zorder=1)
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, zorder=0)

    for idx, (label, x, mean, std) in enumerate(uq_curves):
        color = cfg.color(idx)
        ax.plot(
            x,
            mean,
            label=label,
            color=color,
            marker="o",
            markersize=3,
            linewidth=cfg.line_width,
            zorder=2,
        )
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=cfg.fill_alpha, zorder=1)

    ax.set_xlabel("Labeled samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        f"Active learning on {dataset_display(dataset)}  $\\mathrm{{({NOTION_LABEL[notion]})}}$",
    )
    ax.grid(visible=True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha, color=cfg.color_gridline)
    ax.set_axisbelow(True)

    # Baselines first with margin before random, without changing the drawing order or the greys.
    handles, _ = ax.get_legend_handles_labels()
    legend_rank = ("base + margin", "base + random")
    handles.sort(
        key=lambda h: next(
            (rank for rank, prefix in enumerate(legend_rank) if h.get_label().startswith(prefix)),
            len(legend_rank),
        ),
    )
    ax.legend(handles=handles, fontsize="small", loc="lower right", ncol=LEGEND_NCOL, framealpha=0.9)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"al_curves{FILENAME_SUFFIX}_{dataset}_{notion}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({len(uq_curves)} UQ + {len(baseline_curves)} baselines)")


def main() -> None:
    for dataset in DATASETS:
        for notion in NOTIONS:
            if (dataset, notion) not in SKIPPED_PANELS:
                _render_panel(dataset, notion)


if __name__ == "__main__":
    main()
