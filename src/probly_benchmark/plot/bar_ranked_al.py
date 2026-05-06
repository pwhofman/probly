r"""Active-learning ranked bar plot.

Three PDFs per ID dataset, one per uncertainty notion (epistemic / aleatoric
/ total). Each PDF:

- **Bars** -- every UQ method from ``cfg.methods`` (excluding the baseline
  ``base`` method) using ``uq_strategy`` (default ``"uncertainty"``) at the
  plot's notion. Methods on the x-axis sorted descending by NAUC; mean and
  std are computed across seeds.
- **Horizontal lines** -- the ``base`` method evaluated under each entry of
  ``baselines.strategies`` (default random / least_confident / margin) with
  ``baselines.supervised_loss = cross_entropy`` and ``baselines.calibration =
  none``. Each line's value is the mean NAUC across seeds; a shaded band
  shows :math:`\pm` one std.

Sidecars: ``bar_al_<dataset>_<notion>.ranking.json`` -- methods ranked by
NAUC under that notion, used by :mod:`probly_benchmark.plot.al_curves_topk`
and :mod:`probly_benchmark.plot.paper_tables`.

Usage::

    uv run bar_ranked_al.py comparison=al_openml_all_methods \
        wandb.project=al_openml_v1600_0505
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np

from probly.plot.config import PlotConfig
from probly_benchmark.plot import cache_al
from probly_benchmark.plot.utils import resolve_label, resolve_save_path

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from omegaconf import DictConfig


_BASELINE_LINESTYLES = ("--", "-.", ":", (0, (3, 1, 1, 1)))
_NOTION_TITLE = {"epistemic": "Epistemic (EU)", "aleatoric": "Aleatoric (AU)", "total": "Total (TU)"}


def _aggregate_nauc(records: list[dict]) -> tuple[float, float] | None:
    """Mean and std of NAUC across seeds, or ``None`` if no records have it."""
    values = []
    for r in records:
        v = r.get("summary", {}).get("nauc")
        if v is None:
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std()) if arr.size > 1 else 0.0


def _write_ranking_json(path: Path, ranking: list[dict]) -> None:
    serializable = [
        {
            "method": e["method"],
            "label": e["label"],
            "mean": float(e["mean"]),
            "std": float(e["std"]),
        }
        for e in ranking
    ]
    path.write_text(json.dumps(serializable, indent=2) + "\n")


def _gather_uq(
    cfg: DictConfig,
    *,
    ds_key: str,
    method_entries: list[DictConfig],
    notion: str,
    default_seeds: list[int] | None,
) -> list[dict]:
    """Per-method NAUC entries for the bar plot at one notion (cache-only)."""
    out: list[dict] = []
    for entry in method_entries:
        if entry.name == cfg.baselines.method:
            continue
        records = cache_al.load_runs(
            cfg.cache.entity,
            cfg.cache.sink,
            ds_key,
            entry.name,
            cfg.uq_strategy,
            notion=notion,
            supervised_loss=cfg.supervised_loss,
            calibration=cfg.calibration,
            seeds=default_seeds,
        )
        agg = _aggregate_nauc(records)
        if agg is None:
            continue
        mean, std = agg
        out.append({"method": entry.name, "label": resolve_label(entry), "mean": mean, "std": std})
    out.sort(key=lambda e: e["mean"], reverse=True)
    return out


def _gather_baselines(
    cfg: DictConfig,
    *,
    ds_key: str,
    default_seeds: list[int] | None,
) -> list[dict]:
    """Mean/std NAUC for each baseline strategy (always ``base`` method, cache-only)."""
    bl = cfg.baselines
    out: list[dict] = []
    for strategy in bl.strategies:
        records = cache_al.load_runs(
            cfg.cache.entity,
            cfg.cache.sink,
            ds_key,
            bl.method,
            strategy,
            notion=None,  # ignored for non-uncertainty (cache uses notion_n_a)
            supervised_loss=bl.supervised_loss,
            calibration=bl.calibration,
            seeds=default_seeds,
        )
        agg = _aggregate_nauc(records)
        if agg is None:
            continue
        mean, std = agg
        out.append({"strategy": strategy, "mean": mean, "std": std})
    return out


def _draw_plot(
    *,
    bars: list[dict],
    baselines: list[dict],
    title: str,
) -> Figure:
    """Bar plot for UQ methods + horizontal baseline reference lines."""
    plot_config = PlotConfig()
    if not bars:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.set_title(title + " (no UQ data)")
        ax.set_axis_off()
        return fig

    method_labels = [e["label"] for e in bars]
    means = [e["mean"] for e in bars]
    stds = [e["std"] for e in bars]
    x = np.arange(len(bars))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(bars) + 2.0), 5.0))

    # Baselines first so the bars sit on top of any shaded bands.
    cmap = plt.get_cmap("Greys")
    legend_handles = []
    for idx, baseline in enumerate(baselines):
        ls = _BASELINE_LINESTYLES[idx % len(_BASELINE_LINESTYLES)]
        color = cmap(0.45 + 0.15 * idx)
        line = ax.axhline(baseline["mean"], linestyle=ls, color=color, linewidth=1.6, zorder=1)
        if baseline["std"] > 0:
            ax.axhspan(
                baseline["mean"] - baseline["std"],
                baseline["mean"] + baseline["std"],
                color=color,
                alpha=0.10,
                zorder=0,
            )
        line.set_label(f"baseline / {baseline['strategy'].replace('_', ' ')}")
        legend_handles.append(line)

    bar_width = 0.8
    bar_handles = ax.bar(
        x,
        means,
        width=bar_width,
        yerr=stds,
        capsize=4,
        color=plot_config.color(0),
        edgecolor="none",
        error_kw={"ecolor": "#333333", "elinewidth": 1.2, "capthick": 1.2},
        zorder=2,
    )
    bar_handles.set_label("UQ method")

    # Side padding equals the inter-bar gap (matches bar_ranked.py).
    inter_bar_gap = 1.0 - bar_width
    ax.set_xlim(-bar_width / 2 - inter_bar_gap, len(bars) - 1 + bar_width / 2 + inter_bar_gap)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=30, ha="right")
    ax.set_ylabel("NAUC")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(
        visible=True,
        linestyle=plot_config.grid_linestyle,
        alpha=plot_config.grid_alpha,
        color=plot_config.color_gridline,
    )
    ax.set_axisbelow(True)
    ax.legend(
        handles=[bar_handles, *legend_handles],
        fontsize="small",
        loc="lower right",
    )
    fig.tight_layout()
    return fig


def _per_dataset_notion(
    cfg: DictConfig,
    ds_key: str,
    notion: str,
    method_entries: list[DictConfig],
    out_dir: Path,
    default_seeds: list[int] | None,
) -> None:
    """Render one bar PDF + ranking JSON for ``(ds_key, notion)``."""
    bars = _gather_uq(
        cfg,
        ds_key=ds_key,
        method_entries=method_entries,
        notion=notion,
        default_seeds=default_seeds,
    )
    baselines = _gather_baselines(
        cfg,
        ds_key=ds_key,
        default_seeds=default_seeds,
    )

    if not bars and not baselines:
        warnings.warn(f"No AL data for {ds_key} / {notion}; skipping.", stacklevel=2)
        return

    title = f"AL NAUC on {ds_key}  -  {_NOTION_TITLE.get(notion, notion)}"
    fig = _draw_plot(bars=bars, baselines=baselines, title=title)
    pdf_path = out_dir / f"bar_al_{ds_key}_{notion}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    _write_ranking_json(out_dir / f"bar_al_{ds_key}_{notion}.ranking.json", bars)
    print(f"Wrote {pdf_path}  ({len(bars)} UQ bars + {len(baselines)} baseline lines)")


@hydra.main(version_base=None, config_path="../plot_configs", config_name="bar_ranked_al")
def main(cfg: DictConfig) -> None:
    """Render per-notion AL bar plots with baseline reference lines."""
    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None
    method_entries = cast("list[DictConfig]", cfg.methods)

    for ds_key in cfg.datasets:
        for notion in cfg.notions:
            _per_dataset_notion(
                cfg,
                ds_key,
                notion,
                method_entries,
                out_dir,
                default_seeds,
            )

    if cfg.get("show", False):
        plt.show()


if __name__ == "__main__":
    main()
