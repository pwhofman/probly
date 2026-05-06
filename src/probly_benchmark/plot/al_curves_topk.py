r"""Top-K AL learning curves per (dataset, notion).

For each ``(dataset, notion)`` pair, looks up the per-notion ranking JSON
written by :mod:`probly_benchmark.plot.bar_ranked_al`, picks the top-K UQ
methods, and overlays one curve per method on a single figure. Each curve is
the mean ``test_accuracy`` across seeds with a shaded :math:`\pm` std band.
Baselines (default ``base`` method x random/least_confident/margin with
cross_entropy + cal=none) are drawn as dashed reference curves.

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

from probly.plot.config import PlotConfig
from probly_benchmark.plot import cache_al
from probly_benchmark.plot.utils import resolve_label, resolve_save_path

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure


_NOTION_TITLE = {"epistemic": "Epistemic (EU)", "aleatoric": "Aleatoric (AU)", "total": "Total (TU)"}


def _load_ranking(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return cast("list[dict]", json.load(fh))


def _select_top_k(
    cfg_methods: list[DictConfig],
    ranking: list[dict] | None,
    top_k: int,
    exclude: set[str],
) -> list[DictConfig]:
    if ranking is None:
        warnings.warn(
            "No AL ranking JSON found; falling back to cfg.methods order. Run bar_ranked_al.py first.",
            stacklevel=2,
        )
        return [e for e in cfg_methods if e.name not in exclude][:top_k]
    by_name = {entry.name: entry for entry in cfg_methods}
    selected: list[DictConfig] = []
    for item in ranking:
        if len(selected) >= top_k:
            break
        if item["method"] in exclude:
            continue
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
    uq_curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    baseline_curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
) -> Figure:
    """Render one ``(dataset, notion)`` figure with overlaid top-K + baselines."""
    plot_config = PlotConfig()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    grey = plt.get_cmap("Greys")

    for idx, (label, x, mean, std) in enumerate(baseline_curves):
        color = grey(0.45 + 0.15 * idx)
        ax.plot(x, mean, label=label, color=color, linestyle="--", linewidth=1.4, zorder=1)
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, zorder=0)

    for idx, (label, x, mean, std) in enumerate(uq_curves):
        color = plot_config.color(idx)
        ax.plot(
            x,
            mean,
            label=label,
            color=color,
            marker="o",
            markersize=3,
            linewidth=plot_config.line_width,
            zorder=2,
        )
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=plot_config.fill_alpha, zorder=1)

    ax.set_xlabel("Labeled samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.grid(
        visible=True,
        linestyle=plot_config.grid_linestyle,
        alpha=plot_config.grid_alpha,
        color=plot_config.color_gridline,
    )
    ax.set_axisbelow(True)
    ax.legend(fontsize="small", loc="lower right")
    fig.tight_layout()
    return fig


def _gather_baseline_curves(
    cfg: DictConfig,
    *,
    ds_key: str,
    cache_mode: str,
    default_seeds: list[int] | None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    bl = cfg.baselines
    out: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for strategy in bl.get("strategies", []):
        try:
            records = cache_al.fetch_with_cache(
                cfg.wandb.entity,
                cfg.wandb.project,
                ds_key=ds_key,
                method_entry={"name": bl.method},
                strategy=strategy,
                notion=None,
                supervised_loss=bl.supervised_loss,
                calibration=bl.calibration,
                default_seeds=default_seeds,
                mode=cache_mode,
            )
        except (RuntimeError, ValueError) as exc:
            warnings.warn(f"Baseline {bl.method}/{strategy} skipped: {exc}", stacklevel=2)
            continue
        arrays = _curve_arrays(records)
        if arrays is None:
            continue
        x, mean, std = arrays
        out.append((f"baseline / {strategy.replace('_', ' ')}", x, mean, std))
    return out


@hydra.main(version_base=None, config_path="../plot_configs", config_name="al_curves_topk")
def main(cfg: DictConfig) -> list[Path]:
    """Produce one top-K AL curve PDF per (dataset, notion)."""
    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(cfg.get("top_k", 4))
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None
    method_entries = cast("list[DictConfig]", cfg.methods)
    notions = list(cfg.notions)
    written: list[Path] = []
    baseline_method = cfg.baselines.method

    for ds_key in cfg.datasets:
        baseline_curves = _gather_baseline_curves(
            cfg,
            ds_key=ds_key,
            cache_mode=cache_mode,
            default_seeds=default_seeds,
        )
        for notion in notions:
            ranking = _load_ranking(out_dir / f"bar_al_{ds_key}_{notion}.ranking.json")
            selected = _select_top_k(method_entries, ranking, top_k, exclude={baseline_method})
            uq_curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            for entry in selected:
                try:
                    records = cache_al.fetch_with_cache(
                        cfg.wandb.entity,
                        cfg.wandb.project,
                        ds_key=ds_key,
                        method_entry=entry,
                        strategy=cfg.uq_strategy,
                        notion=notion,
                        supervised_loss=cfg.supervised_loss,
                        calibration=cfg.calibration,
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
                uq_curves.append((resolve_label(entry), x, mean, std))

            if not uq_curves and not baseline_curves:
                warnings.warn(f"No AL curves for {ds_key} / {notion}; skipping.", stacklevel=2)
                continue

            fig = _draw_curves(
                uq_curves=uq_curves,
                baseline_curves=baseline_curves,
                title=f"AL on {ds_key}  -  {_NOTION_TITLE.get(notion, notion)} (top-{len(uq_curves)})",
            )
            out_path = out_dir / f"al_curves_topk_{ds_key}_{notion}.pdf"
            fig.savefig(out_path)
            plt.close(fig)
            written.append(out_path)
            print(f"Wrote {out_path}  ({len(uq_curves)} UQ + {len(baseline_curves)} baselines)")

    if cfg.get("show", False):
        plt.show()

    return written


if __name__ == "__main__":
    main()
