"""ID-vs-OOD score histograms, one PDF per (target, method, ood-dataset).

Each PDF is a single panel comparing the target's ID scores against one OOD
dataset's scores for one method. Seeds are concatenated. Reads only the local
cache; run ``python -m paper_plots.fetch_ood`` first to materialize score
arrays from the ``id_scores-...`` / ``ood_scores-...`` artifacts. Then::

    uv run python -m paper_plots.ood_hist

Expect a lot of PDFs (datasets x methods x near+far OOD datasets).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import matplotlib.pyplot as plt
import numpy as np

from paper_plots import cache, ood_groups
from paper_plots.metadata import EXCLUDED_METHODS, dataset_display, method_label
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import PlotConfig
from paper_plots.targets import ENTITY, TARGETS, Target

if TYPE_CHECKING:
    from pathlib import Path

MEASURE = "default"
DECOMPOSITION = "epistemic"
BINS = 50
# Per-panel size shared with the other "squarish" plots; the figure scales
# horizontally with the number of OOD subplots.
FIGSIZE: tuple[float, float] = (4.5, 3.6)


def _stack_arrays(records: list[dict], key: str) -> np.ndarray:
    """Concatenate seed score arrays for ``key``; ``[]`` when none are usable."""
    arrays: list[np.ndarray] = []
    for r in records:
        value = r["summary"].get(key)
        if value is None or cache.is_large_array_placeholder(value):
            continue
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            continue
        arrays.append(arr)
    return np.concatenate(arrays) if arrays else np.empty((0,), dtype=float)


def _render_panel(
    target: Target,
    method: str,
    ood_ds: str,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    out_dir: Path,
) -> None:
    """Single (dataset, method, ood_dataset) histogram -> one PDF."""
    cfg = PlotConfig()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(
        id_scores,
        bins=BINS,
        alpha=cfg.histogram_alpha,
        density=True,
        color=cfg.color_negative,
        label=dataset_display(target.dataset),
    )
    ax.hist(
        ood_scores,
        bins=BINS,
        alpha=cfg.histogram_alpha,
        density=True,
        color=cfg.color_positive,
        label=dataset_display(ood_ds),
    )
    ax.set_xlabel("Uncertainty score")
    ax.set_title(f"{method_label(method)}: {dataset_display(target.dataset)} vs {dataset_display(ood_ds)}")
    ax.grid(visible=True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha, color=cfg.color_gridline)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = out_dir / f"ood_hist_{target.dataset}_{method}_{ood_ds}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _render_method(
    target: Target,
    method: str,
    records: list[dict],
    ood_datasets: tuple[str, ...],
    out_dir: Path,
) -> None:
    id_key = f"ood/{MEASURE}/{DECOMPOSITION}/id_scores"
    id_scores = _stack_arrays(records, id_key)
    if id_scores.size == 0:
        return
    for ood_ds in ood_datasets:
        ood_key = f"ood/{ood_ds}/{MEASURE}/{DECOMPOSITION}/ood_scores"
        ood_scores = _stack_arrays(records, ood_key)
        if ood_scores.size == 0:
            continue
        _render_panel(target, method, ood_ds, id_scores, ood_scores, out_dir)


def _render_one(target: Target) -> None:
    methods = [
        m
        for m in cache.list_methods(ENTITY, target.project, target.dataset, target.base_model)
        if m not in EXCLUDED_METHODS
    ]
    if not methods:
        warnings.warn(f"no cached OOD runs for {target.dataset}/{target.base_model}", stacklevel=2)
        return
    ood_datasets = ood_groups.near_ood_for(target.dataset) + ood_groups.far_ood_for(target.dataset)
    out_dir = PAPER_ARTIFACTS / f"{target.dataset}_methods" / "ood_hist"
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for method in methods:
        records = cache.load_records(ENTITY, target.project, target.dataset, target.base_model, method)
        if not records:
            continue
        _render_method(target, method, records, ood_datasets, out_dir)
        rendered += 1
    if rendered == 0:
        warnings.warn(
            f"no score arrays cached for {target.dataset}; check that id_scores / ood_scores artifacts exist",
            stacklevel=2,
        )


def main() -> None:
    for target in TARGETS:
        _render_one(target)


if __name__ == "__main__":
    main()
