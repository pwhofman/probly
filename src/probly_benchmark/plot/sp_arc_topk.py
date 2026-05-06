r"""Top-K accuracy-rejection curves for selective prediction.

Renders accuracy-rejection curves for only the top-K methods according to a
ranking written by ``bar_ranked.py task=sp``. Reads
``<save_path>/bar_sp_<dataset>.ranking.json``; if that file is missing, falls
back to the order of ``cfg.methods``.

Usage::

    uv run sp_arc_topk.py comparison=cifar10_all_methods \
        wandb.project=cifar10-benchmark top_k=4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly.plot.config import PlotConfig
from probly_benchmark.plot.utils import fetch_sp_runs, resolve_label, resolve_save_path

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _load_ranking(path: Path) -> list[dict] | None:
    """Load a ranking JSON written by ``bar_ranked.py``, or ``None``."""
    if not path.exists():
        return None
    with path.open() as fh:
        return cast("list[dict]", json.load(fh))


def _select_top_k_entries(cfg_methods: list[DictConfig], ranking: list[dict] | None, top_k: int) -> list[DictConfig]:
    """Return the top-K method entries from cfg.methods.

    If a ranking JSON is available, uses its method order. Otherwise falls
    back to the order in ``cfg.methods`` and warns.
    """
    if ranking is None:
        warnings.warn(
            "No ranking JSON found; falling back to cfg.methods order. Run bar_ranked.py task=sp first.",
            stacklevel=2,
        )
        return list(cfg_methods)[:top_k]

    by_name = {entry.name: entry for entry in cfg_methods}
    selected: list[DictConfig] = []
    for item in ranking:
        if len(selected) >= top_k:
            break
        name = item["method"]
        entry = by_name.get(name)
        if entry is None:
            continue
        selected.append(entry)
    return selected


@hydra.main(version_base=None, config_path="../plot_configs", config_name="sp_arc_topk")
def main(cfg: DictConfig) -> Figure:
    """Plot AR curves for the top-K methods.

    Args:
        cfg: Hydra config composed from a ``sp_arc_topk`` comparison config.

    Returns:
        The matplotlib Figure.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model

    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = _load_ranking(out_dir / f"bar_sp_{dataset}.ranking.json")
    top_k = int(cfg.get("top_k", 4))
    selected = _select_top_k_entries(list(cfg.methods), ranking, top_k)

    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")

    plot_config = PlotConfig()
    fig, ax = plt.subplots()
    for idx, entry in enumerate(selected):
        try:
            runs = fetch_sp_runs(
                cfg.wandb.entity,
                cfg.wandb.project,
                entry,
                dataset,
                base_model,
                list(cfg.seeds) if cfg.get("seeds") else None,
                measure=cfg.get("measure", "default"),
                decomposition=cfg.get("decomposition", "total"),
                cache_mode=cache_mode,
            )
        except RuntimeError as exc:
            warnings.warn(f"Skipping {entry.name}: {exc}", stacklevel=2)
            continue

        bin_losses_stack = np.stack([r["bin_losses"] for r in runs])
        mean_acc = 1.0 - bin_losses_stack.mean(axis=0)
        std_acc = bin_losses_stack.std(axis=0)
        auroc_acc = 1.0 - float(np.mean([r["auroc"] for r in runs]))

        n_bins = len(mean_acc)
        rejection_rates = np.linspace(0.0, 1.0, n_bins)
        label = f"{resolve_label(entry)} (Acc-AUC={auroc_acc:.3f})"
        color = plot_config.color(idx)

        ax.plot(rejection_rates, mean_acc, label=label, color=color, linewidth=plot_config.line_width)
        if len(runs) > 1:
            ax.fill_between(
                rejection_rates,
                mean_acc - std_acc,
                mean_acc + std_acc,
                alpha=plot_config.fill_alpha,
                color=color,
                linewidth=0,
            )

    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"Top-{top_k} selective prediction on {dataset}")
    ax.grid(
        visible=True,
        linestyle=plot_config.grid_linestyle,
        alpha=plot_config.grid_alpha,
        color=plot_config.color_gridline,
    )
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / f"sp_arc_topk_{dataset}.pdf"
    fig.savefig(out_path)
    print(f"Wrote {out_path}  (top-{top_k} of {len(selected)} drawn)")

    if cfg.get("show", False):
        plt.show()

    return fig


if __name__ == "__main__":
    main()
