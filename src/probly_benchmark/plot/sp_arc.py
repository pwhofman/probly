"""Accuracy-rejection curve plots for selective prediction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly_benchmark.paths import FIGURE_PATH
from probly_benchmark.plot.utils import fetch_sp_runs

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONFIG_DIR = Path(__file__).parent.parent / "configs"
_METHOD_CONFIG_DIR = _CONFIG_DIR / "method"


def _resolve_label(entry: DictConfig) -> str:
    if entry.get("label"):
        return str(entry.label)
    cfg_path = _METHOD_CONFIG_DIR / f"{entry.name}.yaml"
    if cfg_path.exists():
        raw = OmegaConf.load(cfg_path)
        if isinstance(raw, DictConfig):
            label = raw.get("label") or raw.get("method", DictConfig({})).get("label")
            if label:
                return str(label)
    return str(entry.name)


@hydra.main(version_base=None, config_path="../configs/plot", config_name="sp_arc")
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

    fig, ax = plt.subplots()

    for entry in cfg.methods:
        runs = fetch_sp_runs(
            cfg.wandb.entity,
            cfg.wandb.project,
            entry,
            dataset,
            base_model,
            list(cfg.seeds) if cfg.get("seeds") else None,
        )

        bin_losses_stack = np.stack([r["bin_losses"] for r in runs])
        mean_acc = 1.0 - bin_losses_stack.mean(axis=0)
        std_acc = bin_losses_stack.std(axis=0)
        auroc_acc = 1.0 - float(np.mean([r["auroc"] for r in runs]))

        n_bins = len(mean_acc)
        rejection_rates = np.linspace(0.0, 1.0, n_bins)
        label = f"{_resolve_label(entry)} (AUROC={auroc_acc:.3f})"

        (line,) = ax.plot(rejection_rates, mean_acc, label=label)
        if len(runs) > 1:
            ax.fill_between(
                rejection_rates,
                mean_acc - std_acc,
                mean_acc + std_acc,
                alpha=0.2,
                color=line.get_color(),
            )

    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()

    if cfg.get("filename"):
        FIGURE_PATH.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_PATH / cfg.filename)
    if cfg.get("show", False):
        plt.show()

    return fig


if __name__ == "__main__":
    main()
