r"""Top-K OOD score histograms — one PDF per (method, OOD dataset).

For each band (near / far) the band's ranking JSON written by
``bar_ranked.py task=ood`` decides which K methods to render. For each
(method, OOD dataset) combination with non-empty score arrays a single-panel
histogram is written as its own PDF (easier to inspect than a composite
grid, and missing combos are simply absent from the directory rather than
silently rendered as empty subplots).

Usage::

    uv run ood_hist_topk.py comparison=cifar10_all_methods \
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

from probly.plot.ood import plot_histogram
from probly_benchmark.plot import ood_groups
from probly_benchmark.plot.utils import display_name, fetch_ood_runs, resolve_label, resolve_save_path

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
    """Return the top-K method entries for a band."""
    if ranking is None:
        warnings.warn(
            "No ranking JSON found; falling back to cfg.methods order. Run bar_ranked.py task=ood first.",
            stacklevel=2,
        )
        return list(cfg_methods)[:top_k]

    by_name = {entry.name: entry for entry in cfg_methods}
    selected: list[DictConfig] = []
    for item in ranking:
        if len(selected) >= top_k:
            break
        entry = by_name.get(item["method"])
        if entry is None:
            continue
        selected.append(entry)
    return selected


def _draw_one_histogram(
    *,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    bins: int,
    title: str,
) -> Figure:
    """Render a single ID-vs-OOD score histogram.

    Falls back to a hand-rolled :func:`matplotlib.pyplot.hist` if the project
    helper raises (e.g. on degenerate score arrays).
    """
    try:
        return cast(
            "Figure",
            plot_histogram(id_scores=id_scores, ood_scores=ood_scores, bins=bins, title=title),
        )
    except (ValueError, TypeError) as exc:
        warnings.warn(f"plot_histogram failed ({exc}); falling back to plain matplotlib.", stacklevel=2)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.hist(id_scores, bins=bins, alpha=0.6, density=True, label="ID")
    ax.hist(ood_scores, bins=bins, alpha=0.6, density=True, label="OOD")
    ax.set_xlabel("Uncertainty score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


@hydra.main(version_base=None, config_path="../plot_configs", config_name="ood_hist_topk")
def main(cfg: DictConfig) -> list[Path]:
    """Render one histogram PDF per (band, method, OOD dataset).

    Args:
        cfg: Hydra config composed from an ``ood_hist_topk`` comparison config.

    Returns:
        List of PDF paths that were written.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model

    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(cfg.get("top_k", 4))
    bins = int(cfg.get("bins", 50))
    measure = cfg.get("measure", "default")
    decomposition = cfg.get("decomposition", "epistemic")
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")

    bands: dict[str, tuple[str, ...]] = {
        "near": ood_groups.near_ood_for(dataset),
        "far": ood_groups.far_ood_for(dataset),
    }

    written: list[Path] = []
    skipped: list[tuple[str, str, str, str]] = []  # (band, method, ood_ds, reason)

    for band_name, band_datasets in bands.items():
        ranking = _load_ranking(out_dir / f"bar_ood_{band_name}_{dataset}.ranking.json")
        selected = _select_top_k_entries(list(cfg.methods), ranking, top_k)
        if not selected:
            warnings.warn(f"No methods to draw for {band_name}-OOD; skipping.", stacklevel=2)
            continue

        for entry in selected:
            try:
                runs_by_ds = fetch_ood_runs(
                    cfg.wandb.entity,
                    cfg.wandb.project,
                    entry,
                    dataset,
                    base_model,
                    ood_datasets=list(band_datasets),
                    default_seeds=list(cfg.seeds) if cfg.get("seeds") else None,
                    measure=measure,
                    decomposition=decomposition,
                    cache_mode=cache_mode,
                )
            except RuntimeError as exc:
                warnings.warn(f"Skipping {entry.name}: {exc}", stacklevel=2)
                skipped.extend((band_name, entry.name, ood_ds, "no OOD runs") for ood_ds in band_datasets)
                continue

            label = resolve_label(entry)
            for ood_ds in band_datasets:
                runs = runs_by_ds.get(ood_ds, [])
                runs = [r for r in runs if r["ood_scores"].size > 0 and r["id_scores"].size > 0]
                if not runs:
                    skipped.append((band_name, entry.name, ood_ds, "score arrays missing"))
                    continue

                id_scores = np.concatenate([r["id_scores"] for r in runs])
                ood_scores = np.concatenate([r["ood_scores"] for r in runs])
                title = f"{label}: {display_name(dataset)} vs {display_name(ood_ds)}"
                fig = _draw_one_histogram(id_scores=id_scores, ood_scores=ood_scores, bins=bins, title=title)

                out_path = out_dir / f"ood_hist_{dataset}_{band_name}_{ood_ds}_{entry.name}.pdf"
                fig.savefig(out_path)
                plt.close(fig)
                written.append(out_path)

    print(f"Wrote {len(written)} histogram PDF(s) to {out_dir}")
    if skipped:
        by_reason: dict[str, int] = {}
        for *_unused, reason in skipped:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        print(f"Skipped {len(skipped)} (method, ood_dataset) pair(s): {by_reason}")

    if cfg.get("show", False):
        plt.show()

    return written


if __name__ == "__main__":
    main()
