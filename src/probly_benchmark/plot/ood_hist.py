"""Uncertainty score histogram plots for OOD detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly.plot.ood import plot_histogram
from probly_benchmark.paths import FIGURE_PATH
from probly_benchmark.plot.utils import fetch_ood_runs, resolve_label

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


@hydra.main(version_base=None, config_path="../plot_configs", config_name="ood_hist")
def main(cfg: DictConfig) -> list[Figure]:
    """Plot uncertainty score histograms for OOD detection methods.

    For each method defined in the config, all matching seeds are concatenated
    into a single histogram showing the in-distribution vs out-of-distribution
    score overlap. One figure is produced per method.

    Args:
        cfg: Hydra config composed from an ``ood_hist`` comparison config.

    Returns:
        List of matplotlib Figures, one per method.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    ood_detection_defaults_raw = OmegaConf.load(_CONFIG_DIR / "ood_detection.yaml")
    ood_detection_defaults = (
        ood_detection_defaults_raw if isinstance(ood_detection_defaults_raw, DictConfig) else DictConfig({})
    )
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model
    ood_dataset: str = (
        cfg.get("ood_dataset") or recipe.get("ood_dataset") or ood_detection_defaults.get("ood_dataset", "")
    )

    bins: int = cfg.get("bins", 50)
    figures: list[Figure] = []

    for entry in cfg.methods:
        runs = fetch_ood_runs(
            cfg.wandb.entity,
            cfg.wandb.project,
            entry,
            dataset,
            ood_dataset,
            base_model,
            list(cfg.seeds) if cfg.get("seeds") else None,
        )

        id_scores = np.concatenate([r["id_scores"] for r in runs])
        ood_scores = np.concatenate([r["ood_scores"] for r in runs])

        label = resolve_label(entry)
        fig = cast(
            "Figure",
            plot_histogram(
                id_scores=id_scores,
                ood_scores=ood_scores,
                bins=bins,
                title=f"{label} - Score Distribution ({dataset} vs {ood_dataset})",
            ),
        )
        figures.append(fig)

        if cfg.get("filename_prefix"):
            FIGURE_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURE_PATH / f"{cfg.filename_prefix}_{entry.name}.pdf")

    if cfg.get("show", False):
        plt.show()

    return figures


if __name__ == "__main__":
    main()
