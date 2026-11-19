"""Per-method ranking aggregation, computed once and shared by bars and tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from paper_plots import cache, ood_groups
from paper_plots.metadata import EXCLUDED_METHODS, method_label
from paper_plots.targets import ENTITY

if TYPE_CHECKING:
    from paper_plots.targets import Target


def sp_ranking(
    target: Target,
    *,
    measure: str = "default",
    decomposition: str = "total",
    seeds: list[int] | None = None,
) -> list[dict]:
    """Per-method SP Acc-AUC ranking sorted descending. ``[]`` if no cache hit."""
    auroc_key = f"sp/{measure}/{decomposition}/auroc"
    methods = cache.list_methods(ENTITY, target.project, target.dataset, target.base_model)
    ranking: list[dict] = []
    for method in methods:
        if method in EXCLUDED_METHODS:
            continue
        records = cache.load_records(ENTITY, target.project, target.dataset, target.base_model, method, seeds)
        aurocs = [r["summary"][auroc_key] for r in records if auroc_key in r["summary"]]
        if not aurocs:
            continue
        accuracy_auc = 1.0 - np.asarray(aurocs, dtype=float)
        ranking.append(
            {
                "method": method,
                "label": method_label(method),
                "mean": float(accuracy_auc.mean()),
                "std": float(accuracy_auc.std()) if accuracy_auc.size > 1 else 0.0,
            }
        )
    ranking.sort(key=lambda r: r["mean"], reverse=True)
    return ranking


def ood_band_ranking(
    target: Target,
    band: str,
    *,
    measure: str = "default",
    decomposition: str = "epistemic",
    seeds: list[int] | None = None,
) -> list[dict]:
    """Per-method OOD AUROC ranking for ``band`` (``"near"`` or ``"far"``).

    Score = mean over the band's OOD datasets of the per-dataset seed-mean
    AUROC. Std = mean over the band's OOD datasets of the per-dataset seed-std.
    """
    if band == "near":
        band_datasets = ood_groups.near_ood_for(target.dataset)
    elif band == "far":
        band_datasets = ood_groups.far_ood_for(target.dataset)
    else:
        msg = f"unknown OOD band {band!r}; expected 'near' or 'far'"
        raise ValueError(msg)

    methods = cache.list_methods(ENTITY, target.project, target.dataset, target.base_model)
    ranking: list[dict] = []
    for method in methods:
        if method in EXCLUDED_METHODS:
            continue
        records = cache.load_records(ENTITY, target.project, target.dataset, target.base_model, method, seeds)
        if not records:
            continue
        per_dataset_means: list[float] = []
        per_dataset_stds: list[float] = []
        for ood_ds in band_datasets:
            key = f"ood/{ood_ds}/{measure}/{decomposition}/auroc"
            aurocs = [r["summary"][key] for r in records if key in r["summary"]]
            if not aurocs:
                continue
            arr = np.asarray(aurocs, dtype=float)
            per_dataset_means.append(float(arr.mean()))
            per_dataset_stds.append(float(arr.std()) if arr.size > 1 else 0.0)
        if not per_dataset_means:
            continue
        ranking.append(
            {
                "method": method,
                "label": method_label(method),
                "mean": float(np.mean(per_dataset_means)),
                "std": float(np.mean(per_dataset_stds)),
            }
        )
    ranking.sort(key=lambda r: r["mean"], reverse=True)
    return ranking
