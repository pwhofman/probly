r"""OOD score histograms for every (method, OOD dataset, seed) with data.

Walks every method in the comparison config, every OOD dataset (near + far),
every seed; writes one histogram PDF per seed that has non-empty score
arrays. When more than one seed survives for a given (method, ood_dataset),
each seed gets its own PDF with a ``_seed{n}`` suffix; if only one survives,
no suffix is added.

Usage::

    uv run ood_hist_all.py comparison=cifar10_all_methods \
        wandb.project=cifar10-benchmark
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly.plot.ood import plot_histogram
from probly_benchmark.plot import ood_groups
from probly_benchmark.plot.utils import fetch_ood_runs, resolve_label, resolve_save_path

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _draw_one_histogram(
    *,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    bins: int,
    title: str,
) -> Figure:
    """Render a single ID-vs-OOD histogram, falling back to plain matplotlib."""
    try:
        return cast(
            "Figure",
            plot_histogram(id_scores=id_scores, ood_scores=ood_scores, bins=bins, title=title),
        )
    except (ValueError, TypeError) as exc:
        warnings.warn(f"plot_histogram failed ({exc}); using fallback.", stacklevel=2)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.hist(id_scores, bins=bins, alpha=0.6, density=True, label="ID")
    ax.hist(ood_scores, bins=bins, alpha=0.6, density=True, label="OOD")
    ax.set_xlabel("Uncertainty score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


@hydra.main(version_base=None, config_path="../plot_configs", config_name="ood_hist_all")
def main(cfg: DictConfig) -> list[Path]:
    """Produce one histogram PDF per surviving (method, OOD dataset, seed).

    Args:
        cfg: Hydra config composed from an ``ood_hist_all`` comparison config.

    Returns:
        List of PDF paths that were written.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model

    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = int(cfg.get("bins", 50))
    measure = cfg.get("measure", "default")
    decomposition = cfg.get("decomposition", "epistemic")
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None

    ood_datasets_by_band: dict[str, tuple[str, ...]] = {
        "near": ood_groups.near_ood_for(dataset),
        "far": ood_groups.far_ood_for(dataset),
    }
    band_for_ds: dict[str, str] = {ds: band for band, datasets in ood_datasets_by_band.items() for ds in datasets}
    all_ood = tuple(band_for_ds)

    written: list[Path] = []
    skipped: dict[str, int] = {}

    for entry in cast("list[DictConfig]", cfg.methods):
        try:
            runs_by_ds = fetch_ood_runs(
                cfg.wandb.entity,
                cfg.wandb.project,
                entry,
                dataset,
                base_model,
                ood_datasets=list(all_ood),
                default_seeds=default_seeds,
                measure=measure,
                decomposition=decomposition,
                cache_mode=cache_mode,
            )
        except RuntimeError as exc:
            warnings.warn(f"Skipping {entry.name}: {exc}", stacklevel=2)
            continue

        label = resolve_label(entry)
        for ood_ds, runs in runs_by_ds.items():
            survivors = [r for r in runs if r["ood_scores"].size > 0 and r["id_scores"].size > 0]
            if not survivors:
                skipped[ood_ds] = skipped.get(ood_ds, 0) + 1
                continue

            multiple = len(survivors) > 1
            band = band_for_ds.get(ood_ds, "unknown")
            for run in survivors:
                seed = run.get("seed")
                title = f"{label}: {dataset} vs {ood_ds}"
                if multiple:
                    title += f"  (seed {seed})"
                fig = _draw_one_histogram(
                    id_scores=np.asarray(run["id_scores"]),
                    ood_scores=np.asarray(run["ood_scores"]),
                    bins=bins,
                    title=title,
                )
                stem = f"ood_hist_{dataset}_{band}_{ood_ds}_{entry.name}"
                if multiple:
                    stem += f"_seed{seed}"
                out_path = out_dir / f"{stem}.pdf"
                fig.savefig(out_path)
                plt.close(fig)
                written.append(out_path)

    print(f"Wrote {len(written)} histogram PDF(s) to {out_dir}")
    if skipped:
        total = sum(skipped.values())
        print(f"Skipped {total} (method, ood_dataset) pair(s) with no surviving score arrays.")

    if cfg.get("show", False):
        plt.show()

    return written


if __name__ == "__main__":
    main()
