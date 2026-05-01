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
from probly_benchmark.plot.utils import fetch_ood_runs

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
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model
    ood_dataset: str = cfg.get("ood_dataset") or recipe.get("ood_dataset", "")

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

        label = _resolve_label(entry)
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

        if cfg.get("filename"):
            FIGURE_PATH.mkdir(parents=True, exist_ok=True)
            stem = Path(cfg.filename).stem
            suffix = Path(cfg.filename).suffix
            fig.savefig(FIGURE_PATH / f"{stem}_{entry.name}{suffix}")

    if cfg.get("show", False):
        plt.show()

    return figures


if __name__ == "__main__":
    main()
