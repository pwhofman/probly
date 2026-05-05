r"""Top-K AL learning curves per (dataset, strategy).

For each ``(dataset, strategy)`` pair, looks up the per-strategy ranking
JSON written by :mod:`probly_benchmark.plot.bar_ranked_al`, picks the top-K
methods, and overlays one curve per method on a single figure. Each curve is
the mean ``test_accuracy`` across seeds with a shaded :math:`\pm` std band.

Usage::

    uv run al_curves_topk.py comparison=al_openml_all_methods \
        wandb.project=al_openml_v1600_0505 top_k=4
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


def _load_ranking(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return cast("list[dict]", json.load(fh))


def _select_top_k(
    cfg_methods: list[DictConfig],
    ranking: list[dict] | None,
    top_k: int,
) -> list[DictConfig]:
    if ranking is None:
        warnings.warn(
            "No AL ranking JSON found; falling back to cfg.methods order. Run bar_ranked_al.py first.",
            stacklevel=2,
        )
        return list(cfg_methods)[:top_k]
    by_name = {entry.name: entry for entry in cfg_methods}
    selected: list[DictConfig] = []
    for item in ranking:
        if len(selected) >= top_k:
            break
        entry = by_name.get(item["method"])
        if entry is not None:
            selected.append(entry)
    return selected


def _curve_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Stack history rows across seeds into ``(x, mean, std)`` numpy arrays."""
    by_seed: list[list[dict]] = []
    for r in records:
        history = r.get("history", [])
        if history:
            by_seed.append(history)
    if not by_seed:
        return None

    # Use the first run's labeled_size axis as the canonical x; assume all
    # runs share the same AL schedule. Drop seeds whose lengths differ.
    expected_len = len(by_seed[0])
    aligned = [h for h in by_seed if len(h) == expected_len]
    if not aligned:
        return None

    x = np.asarray([row["labeled_size"] for row in aligned[0]], dtype=float)
    accuracy = np.asarray([[row["test_accuracy"] for row in h] for h in aligned], dtype=float)
    mean = accuracy.mean(axis=0)
    std = accuracy.std(axis=0) if accuracy.shape[0] > 1 else np.zeros_like(mean)
    return x, mean, std


def _draw_curves(
    *,
    curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
) -> Figure:
    """Render one ``(dataset, strategy)`` figure with overlaid top-K curves."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("tab10")
    for idx, (label, x, mean, std) in enumerate(curves):
        color = cmap(idx % 10)
        (line,) = ax.plot(x, mean, label=label, color=color, marker="o", markersize=3)
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("Labeled samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    ax.legend(fontsize="small", loc="lower right")
    fig.tight_layout()
    return fig


@hydra.main(version_base=None, config_path="../plot_configs", config_name="al_curves_topk")
def main(cfg: DictConfig) -> list[Path]:
    """Produce one top-K AL curve PDF per (dataset, strategy)."""
    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(cfg.get("top_k", 4))
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None
    method_entries = cast("list[DictConfig]", cfg.methods)
    strategies = list(cfg.strategies)
    written: list[Path] = []

    for ds_key in cfg.datasets:
        for strategy in strategies:
            ranking = _load_ranking(out_dir / f"bar_al_{ds_key}_{strategy}.ranking.json")
            selected = _select_top_k(method_entries, ranking, top_k)
            if not selected:
                continue

            curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            for entry in selected:
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
                    warnings.warn(f"Skipping {entry.name}: {exc}", stacklevel=2)
                    continue
                arrays = _curve_arrays(records)
                if arrays is None:
                    continue
                x, mean, std = arrays
                curves.append((resolve_label(entry), x, mean, std))

            if not curves:
                warnings.warn(
                    f"No AL curves for {ds_key} / {strategy}; skipping.",
                    stacklevel=2,
                )
                continue

            fig = _draw_curves(
                curves=curves,
                title=f"AL on {ds_key}  ({strategy}, top-{len(curves)})",
            )
            out_path = out_dir / f"al_curves_topk_{ds_key}_{strategy}.pdf"
            fig.savefig(out_path)
            plt.close(fig)
            written.append(out_path)
            print(f"Wrote {out_path}  ({len(curves)} curves)")

    if cfg.get("show", False):
        plt.show()

    return written


if __name__ == "__main__":
    main()
