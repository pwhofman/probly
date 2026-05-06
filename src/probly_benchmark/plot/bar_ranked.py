r"""Ranked bar plots across methods for selective prediction and OOD detection.

Each invocation produces one or more bar plots ordered descending by score
(taller = better) plus a sidecar ``*.ranking.json`` listing methods in
ranked order. The JSON drives the top-K detail figures so the bar order and
the detail figure stay in sync.

Usage::

    uv run bar_ranked.py task=sp comparison=cifar10_all_methods \
        wandb.project=cifar10-benchmark
    uv run bar_ranked.py task=ood comparison=cifar10_all_methods \
        wandb.project=cifar10-benchmark
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
from probly_benchmark.plot import ood_groups
from probly_benchmark.plot.utils import fetch_ood_runs, fetch_sp_runs, resolve_label, resolve_save_path

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _draw_bars(
    ranking: list[dict],
    *,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> Figure:
    """Render a descending bar chart from a ranked entry list.

    Args:
        ranking: List of dicts (already sorted descending) with ``label``,
            ``mean`` and ``std``.
        ylabel: Y-axis label for the bars.
        title: Bold main title shown above the axes.
        subtitle: Optional non-bold parenthetical subtitle (e.g.
            ``"(avg over 4 datasets)"``) drawn just below the title.
        ylim: Y-axis limits, or ``None`` to let matplotlib auto-scale.

    Returns:
        The matplotlib Figure.
    """
    plot_config = PlotConfig()
    labels = [entry["label"] for entry in ranking]
    means = [entry["mean"] for entry in ranking]
    stds = [entry["std"] for entry in ranking]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(labels) + 2.0), 4.5))
    bar_width = 0.8  # matplotlib default; explicit so we can size the side padding
    ax.bar(
        x,
        means,
        width=bar_width,
        yerr=stds,
        capsize=4,
        color=plot_config.color(0),
        edgecolor="none",
        error_kw={"ecolor": "#333333", "elinewidth": 1.2, "capthick": 1.2},
    )
    # Tight side margins: padding from the axis edge to the first/last bar
    # equals the gap BETWEEN adjacent bars (= 1 - bar_width = 0.2 for the
    # default), instead of matplotlib's 5% auto-margin which leaves much
    # more empty space.
    inter_bar_gap = 1.0 - bar_width
    ax.set_xlim(-bar_width / 2 - inter_bar_gap, len(labels) - 1 + bar_width / 2 + inter_bar_gap)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if subtitle:
        # Lift the bold title so the regular-weight subtitle has room below.
        ax.set_title(title, pad=18)
        ax.annotate(
            subtitle,
            xy=(0.5, 1.005),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=plot_config.label_fontsize * 0.85,
            fontweight="normal",
            color="#555555",
        )
    else:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(
        visible=True,
        linestyle=plot_config.grid_linestyle,
        alpha=plot_config.grid_alpha,
        color=plot_config.color_gridline,
    )
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _write_ranking_json(path: Path, ranking: list[dict]) -> None:
    """Persist a ranking list as JSON next to a bar PDF."""
    serializable = [
        {
            "method": entry["method"],
            "label": entry["label"],
            "mean": float(entry["mean"]),
            "std": float(entry["std"]),
        }
        for entry in ranking
    ]
    path.write_text(json.dumps(serializable, indent=2) + "\n")


def _run_sp(cfg: DictConfig, dataset: str, base_model: str) -> dict[str, tuple[Figure, list[dict]]]:
    """Build the SP ranked bar plot.

    Score = ``1 - sp/{measure}/{decomp}/auroc`` (accuracy-AUC, higher is
    better). Per-method bars use the seed mean and std.
    """
    measure = cfg.sp.get("measure", "default")
    decomposition = cfg.sp.get("decomposition", "total")
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None

    ranking: list[dict] = []
    for entry in cast("list[DictConfig]", cfg.methods):
        try:
            runs = fetch_sp_runs(
                cfg.wandb.entity,
                cfg.wandb.project,
                entry,
                dataset,
                base_model,
                default_seeds,
                measure=measure,
                decomposition=decomposition,
                cache_mode=cache_mode,
            )
        except RuntimeError as exc:
            warnings.warn(f"Skipping {entry.name}: {exc}", stacklevel=2)
            continue

        accuracy_auc = np.array([1.0 - r["auroc"] for r in runs], dtype=float)
        ranking.append(
            {
                "method": entry.name,
                "label": resolve_label(entry),
                "mean": float(accuracy_auc.mean()),
                "std": float(accuracy_auc.std()) if len(accuracy_auc) > 1 else 0.0,
            }
        )

    ranking.sort(key=lambda r: r["mean"], reverse=True)
    fig = _draw_bars(
        ranking,
        ylabel="Accuracy-AUC (1 - AR-AUROC)",
        title=f"Selective prediction on {dataset}",
    )
    return {"sp": (fig, ranking)}


def _run_ood(cfg: DictConfig, dataset: str, base_model: str) -> dict[str, tuple[Figure, list[dict]]]:
    """Build the near-OOD and far-OOD ranked bar plots.

    Per-method score for a band = mean over the band's OOD datasets of the
    per-dataset seed-mean AUROC. Per-method std = mean over the band's OOD
    datasets of the per-dataset seed-std (so the bar shows the typical
    seed-to-seed variation we see at the dataset level).
    """
    measure = cfg.ood.get("measure", "default")
    decomposition = cfg.ood.get("decomposition", "epistemic")
    cache_mode = cfg.get("cache", DictConfig({})).get("mode", "read")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None

    near = ood_groups.near_ood_for(dataset)
    far = ood_groups.far_ood_for(dataset)
    bands = {"near": near, "far": far}

    rankings: dict[str, list[dict]] = {"near": [], "far": []}

    for entry in cast("list[DictConfig]", cfg.methods):
        try:
            runs_by_ds = fetch_ood_runs(
                cfg.wandb.entity,
                cfg.wandb.project,
                entry,
                dataset,
                base_model,
                ood_datasets=list({*near, *far}),
                default_seeds=default_seeds,
                measure=measure,
                decomposition=decomposition,
                cache_mode=cache_mode,
            )
        except RuntimeError as exc:
            warnings.warn(f"Skipping {entry.name}: {exc}", stacklevel=2)
            continue

        for band_name, band_datasets in bands.items():
            per_dataset_means: list[float] = []
            per_dataset_stds: list[float] = []
            for ood_ds in band_datasets:
                runs = runs_by_ds.get(ood_ds, [])
                aurocs = [r["auroc"] for r in runs if r["auroc"] is not None]
                if not aurocs:
                    continue
                arr = np.asarray(aurocs, dtype=float)
                per_dataset_means.append(float(arr.mean()))
                per_dataset_stds.append(float(arr.std()) if len(arr) > 1 else 0.0)

            if not per_dataset_means:
                warnings.warn(
                    f"No {band_name}-OOD AUROCs for method '{entry.name}' on "
                    f"{dataset}/{base_model}; skipping in this band.",
                    stacklevel=2,
                )
                continue

            rankings[band_name].append(
                {
                    "method": entry.name,
                    "label": resolve_label(entry),
                    "mean": float(np.mean(per_dataset_means)),
                    "std": float(np.mean(per_dataset_stds)),
                }
            )

    out: dict[str, tuple[Figure, list[dict]]] = {}
    for band_name, items in rankings.items():
        items.sort(key=lambda r: r["mean"], reverse=True)
        title_band = "Near-OOD" if band_name == "near" else "Far-OOD"
        fig = _draw_bars(
            items,
            ylabel="OOD AUROC",
            title=f"{title_band} detection on {dataset}",
            subtitle=f"(avg over {len(bands[band_name])} datasets)",
        )
        out[band_name] = (fig, items)
    return out


@hydra.main(version_base=None, config_path="../plot_configs", config_name="bar_ranked")
def main(cfg: DictConfig) -> dict[str, tuple[Figure, list[dict]]]:
    """Produce ranked bar plots and ranking JSONs for SP or OOD.

    Args:
        cfg: Hydra config composed from a ``bar_ranked`` comparison config.
            ``cfg.task`` selects ``"sp"`` or ``"ood"``.

    Returns:
        Dict mapping a band key (``"sp"``, ``"near"``, ``"far"``) to a tuple
        of ``(figure, ranking)``. Useful when imported directly.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model
    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    task = cfg.task
    if task == "sp":
        results = _run_sp(cfg, dataset, base_model)
        stems = {"sp": f"bar_sp_{dataset}"}
    elif task == "ood":
        results = _run_ood(cfg, dataset, base_model)
        stems = {"near": f"bar_ood_near_{dataset}", "far": f"bar_ood_far_{dataset}"}
    else:
        msg = f"Unknown task {task!r}. Expected 'sp' or 'ood'."
        raise ValueError(msg)

    for key, (fig, ranking) in results.items():
        stem = stems[key]
        fig.savefig(out_dir / f"{stem}.pdf")
        _write_ranking_json(out_dir / f"{stem}.ranking.json", ranking)
        print(f"Wrote {out_dir / (stem + '.pdf')}  ({len(ranking)} methods ranked)")

    if cfg.get("show", False):
        plt.show()

    return results


if __name__ == "__main__":
    main()
