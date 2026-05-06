"""Uncertainty score histogram plots for OOD detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly.plot.ood import plot_histogram
from probly_benchmark.plot.utils import display_name, fetch_ood_runs, resolve_label, resolve_save_path

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
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")
    figures: list[Figure] = []

    for entry in cfg.methods:
        try:
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
                cache_mode=cache_mode,
            )
        except RuntimeError as exc:
            warnings.warn(f"Skipping {entry.name}: {exc}", stacklevel=2)
            continue

        # Drop OOD datasets where score arrays are empty (e.g. wandb large-array
        # placeholders that never got their h5 payload uploaded).
        runs_by_ds = {ds: runs for ds, runs in runs_by_ds.items() if any(r["ood_scores"].size > 0 for r in runs)}
        if not runs_by_ds:
            warnings.warn(
                f"Skipping {entry.name}: no OOD score arrays available (only scalar metrics).",
                stacklevel=2,
            )
            continue

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
                    title=f"{label} - Score Distribution ({display_name(dataset)} vs {display_name(ood_ds)})",
                ),
            )
        else:
            fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), squeeze=False)
            for ax, ood_ds in zip(axes[0], available_ds, strict=True):
                runs = runs_by_ds[ood_ds]
                id_scores = np.concatenate([r["id_scores"] for r in runs])
                ood_scores = np.concatenate([r["ood_scores"] for r in runs])
                ax.hist(id_scores, bins=bins, alpha=0.6, density=True, label=display_name(dataset))
                ax.hist(ood_scores, bins=bins, alpha=0.6, density=True, label=display_name(ood_ds))
                ax.set_title(display_name(ood_ds))
                ax.set_xlabel("Uncertainty score")
                ax.legend()
            fig.suptitle(label)
            fig.tight_layout()

        figures.append(fig)

        if cfg.get("filename_prefix"):
            out_dir = resolve_save_path(cfg.get("save_path"))
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{cfg.filename_prefix}_{entry.name}.pdf")

    if cfg.get("show", False):
        plt.show()

    return figures


if __name__ == "__main__":
    main()
