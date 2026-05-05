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

    For each method defined in the config, one figure is produced showing
    the in-distribution vs out-of-distribution score overlap. When multiple
    OOD datasets are available, each dataset is shown as a separate subplot
    within the method's figure. Seeds are concatenated before plotting.

    Args:
        cfg: Hydra config composed from an ``ood_hist`` comparison config.

    Returns:
        List of matplotlib Figures, one per method.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model

    ood_datasets: list[str] | None = list(cfg.ood_datasets) if cfg.get("ood_datasets") else None
    bins: int = cfg.get("bins", 50)
    measure: str = cfg.get("measure", "default")
    decomposition: str = cfg.get("decomposition", "epistemic")
    figures: list[Figure] = []

    for entry in cfg.methods:
        runs_by_ds = fetch_ood_runs(
            cfg.wandb.entity,
            cfg.wandb.project,
            entry,
            dataset,
            base_model,
            ood_datasets=ood_datasets,
            default_seeds=list(cfg.seeds) if cfg.get("seeds") else None,
            measure=measure,
            decomposition=decomposition,
        )

        available_ds = sorted(runs_by_ds)
        n_ds = len(available_ds)
        label = resolve_label(entry)

        if n_ds == 1:
            ood_ds = available_ds[0]
            runs = runs_by_ds[ood_ds]
            id_scores = np.concatenate([r["id_scores"] for r in runs])
            ood_scores = np.concatenate([r["ood_scores"] for r in runs])
            fig = cast(
                "Figure",
                plot_histogram(
                    id_scores=id_scores,
                    ood_scores=ood_scores,
                    bins=bins,
                    title=f"{label} - Score Distribution ({dataset} vs {ood_ds})",
                ),
            )
        else:
            fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), squeeze=False)
            for ax, ood_ds in zip(axes[0], available_ds, strict=True):
                runs = runs_by_ds[ood_ds]
                id_scores = np.concatenate([r["id_scores"] for r in runs])
                ood_scores = np.concatenate([r["ood_scores"] for r in runs])
                ax.hist(id_scores, bins=bins, alpha=0.6, density=True, label=dataset)
                ax.hist(ood_scores, bins=bins, alpha=0.6, density=True, label=ood_ds)
                ax.set_title(ood_ds)
                ax.set_xlabel("Uncertainty score")
                ax.legend()
            fig.suptitle(label)
            fig.tight_layout()

        figures.append(fig)

        if cfg.get("filename_prefix"):
            FIGURE_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURE_PATH / f"{cfg.filename_prefix}_{entry.name}.pdf")

    if cfg.get("show", False):
        plt.show()

    return figures


if __name__ == "__main__":
    main()
