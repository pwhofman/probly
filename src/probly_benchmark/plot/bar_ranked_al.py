r"""Active-learning ranked bar plot.

For each ID dataset, produces one PDF with methods on the x-axis (descending
by mean NAUC across the configured strategies) and one bar per strategy
within each method tick. Mean and std are computed across seeds.

Sidecar JSONs are written alongside each PDF:

- ``bar_al_<dataset>.ranking.json`` -- methods ranked by mean NAUC across
  strategies (the headline ranking shown by the bar order).
- ``bar_al_<dataset>_<strategy>.ranking.json`` -- per-strategy ranking,
  consumed by :mod:`probly_benchmark.plot.al_curves_topk` and
  :mod:`probly_benchmark.plot.paper_tables`.

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
from omegaconf import DictConfig

from probly_benchmark.plot import cache_al
from probly_benchmark.plot.utils import resolve_label, resolve_save_path

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure


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
            "method": entry["method"],
            "label": entry["label"],
            "mean": float(entry["mean"]),
            "std": float(entry["std"]),
        }
        for entry in ranking
    ]
    path.write_text(json.dumps(serializable, indent=2) + "\n")


def _draw_grouped_bars(
    *,
    method_labels: list[str],
    strategies: list[str],
    means: dict[tuple[str, str], float],
    stds: dict[tuple[str, str], float],
    title: str,
) -> Figure:
    """Render one grouped bar chart for a single dataset.

    Methods on the x-axis, one bar per strategy within each method tick.
    """
    n_methods = len(method_labels)
    n_strats = len(strategies)
    bar_width = 0.85 / max(n_strats, 1)
    x = np.arange(n_methods)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * n_methods + 2.0), 5.0))
    for s_idx, strategy in enumerate(strategies):
        offsets = (s_idx - (n_strats - 1) / 2.0) * bar_width
        ms = [means.get((m, strategy), float("nan")) for m in method_labels]
        ss = [stds.get((m, strategy), 0.0) for m in method_labels]
        ax.bar(
            x + offsets,
            ms,
            bar_width,
            yerr=ss,
            capsize=3,
            color=cmap(s_idx % 10),
            label=strategy,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=30, ha="right")
    ax.set_ylabel("NAUC")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(visible=True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(title="Strategy", fontsize="small", ncol=min(n_strats, 3))
    fig.tight_layout()
    return fig


def _per_dataset(
    cfg: DictConfig,
    ds_key: str,
    strategies: list[str],
    method_entries: list[DictConfig],
    out_dir: Path,
    cache_mode: str,
    default_seeds: list[int] | None,
) -> None:
    """Produce one bar PDF + ranking JSONs for a single ID dataset."""
    means: dict[tuple[str, str], float] = {}
    stds: dict[tuple[str, str], float] = {}
    per_strategy: dict[str, list[dict]] = {s: [] for s in strategies}
    seen_methods: list[tuple[str, str]] = []  # (name, label) preserving order

    for entry in method_entries:
        label = resolve_label(entry)
        any_data = False
        for strategy in strategies:
            try:
                records = cache_al.fetch_with_cache(
                    cfg.wandb.entity,
                    cfg.wandb.project,
                    ds_key=ds_key,
                    method_entry=entry,
                    strategy=strategy,
                    default_seeds=default_seeds,
                    mode=cache_mode,
                )
            except (RuntimeError, ValueError) as exc:
                warnings.warn(f"Skipping {entry.name} / {strategy}: {exc}", stacklevel=2)
                continue
            agg = _aggregate_nauc(records)
            if agg is None:
                continue
            any_data = True
            mean, std = agg
            means[(entry.name, strategy)] = mean
            stds[(entry.name, strategy)] = std
            per_strategy[strategy].append({"method": entry.name, "label": label, "mean": mean, "std": std})
        if any_data:
            seen_methods.append((entry.name, label))

    if not seen_methods:
        warnings.warn(f"No AL data for any method on {ds_key}; skipping.", stacklevel=2)
        return

    # Headline ranking: mean NAUC across all strategies (only those present).
    aggregate: list[dict] = []
    for name, label in seen_methods:
        ms = [means[(name, s)] for s in strategies if (name, s) in means]
        ss = [stds[(name, s)] for s in strategies if (name, s) in means]
        if not ms:
            continue
        aggregate.append(
            {
                "method": name,
                "label": label,
                "mean": float(np.mean(ms)),
                "std": float(np.mean(ss)),
            }
        )
    aggregate.sort(key=lambda e: e["mean"], reverse=True)
    method_order_labels = [e["label"] for e in aggregate]
    method_order_names = [e["method"] for e in aggregate]

    fig = _draw_grouped_bars(
        method_labels=method_order_labels,
        strategies=strategies,
        means={(name, s): means[(name, s)] for name in method_order_names for s in strategies if (name, s) in means},
        stds={(name, s): stds[(name, s)] for name in method_order_names for s in strategies if (name, s) in means},
        title=f"AL NAUC on {ds_key}",
    )
    pdf_path = out_dir / f"bar_al_{ds_key}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    _write_ranking_json(out_dir / f"bar_al_{ds_key}.ranking.json", aggregate)
    print(f"Wrote {pdf_path}  ({len(method_order_labels)} methods x {len(strategies)} strategies)")

    # Per-strategy ranking sidecars (consumed by curves + paper_tables).
    for strategy, items in per_strategy.items():
        if not items:
            continue
        items.sort(key=lambda e: e["mean"], reverse=True)
        _write_ranking_json(out_dir / f"bar_al_{ds_key}_{strategy}.ranking.json", items)


@hydra.main(version_base=None, config_path="../plot_configs", config_name="bar_ranked_al")
def main(cfg: DictConfig) -> None:
    """Render grouped-bar AL plots and write per-strategy ranking JSONs."""
    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None
    strategies = list(cfg.strategies)
    method_entries = cast("list[DictConfig]", cfg.methods)

    for ds_key in cfg.datasets:
        _per_dataset(cfg, ds_key, strategies, method_entries, out_dir, cache_mode, default_seeds)

    if cfg.get("show", False):
        plt.show()


if __name__ == "__main__":
    main()
