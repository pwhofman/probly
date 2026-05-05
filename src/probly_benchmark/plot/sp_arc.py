"""Accuracy-rejection curve plots for selective prediction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly.plot.config import PlotConfig
from probly_benchmark.plot.utils import fetch_sp_runs, resolve_label, resolve_save_path

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


@hydra.main(version_base=None, config_path="../plot_configs", config_name="sp_arc")
def main(cfg: DictConfig) -> Figure:
    """Plot accuracy-rejection curves for the methods defined in the config.

    Wandb stores 0/1 error rates (``sp/bin_losses``) and loss-based AUROC
    (``sp/auroc``). Both are flipped to accuracy space before plotting:
    ``accuracy = 1 - loss`` and ``auroc_accuracy = 1 - auroc_loss``.

    When multiple seeds are available for a method, the mean is plotted with a
    shaded +/- one standard deviation band.

    Args:
        cfg: Hydra config composed from an ``sp_arc`` comparison config.

    Returns:
        The matplotlib Figure.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model

    plot_config = PlotConfig()
    fig, ax = plt.subplots()
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")

    for idx, entry in enumerate(cfg.methods):
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

        bin_losses_stack = np.stack([r["bin_losses"] for r in runs])
        mean_acc = 1.0 - bin_losses_stack.mean(axis=0)
        std_acc = bin_losses_stack.std(axis=0)
        auroc_acc = 1.0 - float(np.mean([r["auroc"] for r in runs]))

        n_bins = len(mean_acc)
        rejection_rates = np.linspace(0.0, 1.0, n_bins)
        label = f"{resolve_label(entry)} (AUROC={auroc_acc:.3f})"
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
    ax.grid(
        visible=True,
        linestyle=plot_config.grid_linestyle,
        alpha=plot_config.grid_alpha,
        color=plot_config.color_gridline,
    )
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()

    if cfg.get("filename") and cfg.get("filename_prefix"):
        out_dir = resolve_save_path(cfg.get("save_path"))
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{cfg.filename_prefix}_{cfg.filename}")
    if cfg.get("show", False):
        plt.show()

    return fig


if __name__ == "__main__":
    main()
