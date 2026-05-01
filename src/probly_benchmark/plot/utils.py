"""Shared utilities for probly_benchmark plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import wandb

if TYPE_CHECKING:
    from omegaconf import DictConfig


def fetch_sp_runs(
    entity: str,
    project: str,
    method_entry: DictConfig,
    dataset: str,
    base_model: str,
    default_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Fetch selective prediction results from wandb for one method entry.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        method_entry: One element from ``cfg.methods``. Must have a ``name`` field.
            Optionally ``calibration`` (maps to ``config.calibration.name``),
            and ``seeds`` (overrides ``default_seeds``).
        dataset: Dataset name as stored in the run config (e.g. ``"cifar10"``).
        base_model: Base model name as stored in the run config (e.g. ``"resnet18"``).
        default_seeds: Seeds to filter on when the method entry has no ``seeds``
            field. ``None`` means all available seeds.

    Returns:
        List of dicts, one per matching run, each with keys:

        - ``bin_losses``: ``np.ndarray`` of per-bin 0/1 error rates.
        - ``auroc``: Float AUROC of the loss-rejection curve.
        - ``seed``: Integer seed of the run.

    Raises:
        RuntimeError: If no runs with selective prediction results are found.
    """
    api = wandb.Api()

    filters: dict[str, Any] = {
        "config.method.name": method_entry.name,
        "config.dataset": dataset,
        "config.base_model": base_model,
    }

    seeds = method_entry.get("seeds") or default_seeds
    if seeds is not None:
        filters["config.seed"] = {"$in": list(seeds)}

    if method_entry.get("calibration"):
        filters["config.calibration.name"] = method_entry.calibration

    filters["state"] = "finished"
    runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at")

    seen_seeds: set[Any] = set()
    results = []
    for run in runs:
        seed = run.config.get("seed")
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        bin_losses = run.summary.get("sp/bin_losses")
        auroc = run.summary.get("sp/auroc")
        if bin_losses is None or auroc is None:
            warnings.warn(
                f"Run {run.id} ({run.name}) matched filters but has no selective "
                "prediction results. Was selective_prediction.py run for this run?",
                stacklevel=2,
            )
            continue
        results.append(
            {
                "bin_losses": np.array(bin_losses),
                "auroc": float(auroc),
                "seed": seed,
            }
        )

    if seeds is not None:
        found_seeds = {r["seed"] for r in results}
        for s in seeds:
            if s not in found_seeds:
                warnings.warn(
                    f"No finished run found for method '{method_entry.name}' "
                    f"seed={s} on {dataset}/{base_model} in {entity}/{project}.",
                    stacklevel=2,
                )

    if not results:
        msg = (
            f"No selective prediction results found for method "
            f"'{method_entry.name}' on {dataset}/{base_model} in "
            f"{entity}/{project}. Check that selective_prediction.py has been "
            f"run and that the filters match existing runs."
        )
        raise RuntimeError(msg)

    return results


def fetch_ood_runs(
    entity: str,
    project: str,
    method_entry: DictConfig,
    dataset: str,
    ood_dataset: str,
    base_model: str,
    default_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Fetch OOD detection results from wandb for one method entry.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        method_entry: One element from ``cfg.methods``. Must have a ``name`` field.
            Optionally ``calibration`` (maps to ``config.calibration.name``),
            and ``seeds`` (overrides ``default_seeds``).
        dataset: In-distribution dataset name as stored in the run config.
        ood_dataset: Out-of-distribution dataset name as stored in the run config.
        base_model: Base model name as stored in the run config.
        default_seeds: Seeds to filter on when the method entry has no ``seeds``
            field. ``None`` means all available seeds.

    Returns:
        List of dicts, one per matching run, each with keys:

        - ``id_scores``: ``np.ndarray`` of in-distribution uncertainty scores.
        - ``ood_scores``: ``np.ndarray`` of out-of-distribution uncertainty scores.
        - ``auroc``: Float AUROC, or ``None`` if not logged.
        - ``seed``: Integer seed of the run.

    Raises:
        RuntimeError: If no runs with OOD histogram data are found.
    """
    api = wandb.Api()

    filters: dict[str, Any] = {
        "config.method.name": method_entry.name,
        "config.dataset": dataset,
        "config.ood_dataset": ood_dataset,
        "config.base_model": base_model,
    }

    seeds = method_entry.get("seeds") or default_seeds
    if seeds is not None:
        filters["config.seed"] = {"$in": list(seeds)}

    if method_entry.get("calibration"):
        filters["config.calibration.name"] = method_entry.calibration

    filters["state"] = "finished"
    runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at")

    seen_seeds: set[Any] = set()
    results = []
    for run in runs:
        seed = run.config.get("seed")
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        id_scores = run.summary.get("ood/id_scores")
        ood_scores = run.summary.get("ood/ood_scores")
        if id_scores is None or ood_scores is None:
            warnings.warn(
                f"Run {run.id} ({run.name}) matched filters but has no OOD score "
                "arrays. Was ood_detection.py run for this run?",
                stacklevel=2,
            )
            continue
        results.append(
            {
                "id_scores": np.array(id_scores),
                "ood_scores": np.array(ood_scores),
                "auroc": float(run.summary["ood/auroc"]) if run.summary.get("ood/auroc") is not None else None,
                "seed": seed,
            }
        )

    if seeds is not None:
        found_seeds = {r["seed"] for r in results}
        for s in seeds:
            if s not in found_seeds:
                warnings.warn(
                    f"No finished run found for method '{method_entry.name}' "
                    f"seed={s} on {dataset}/{base_model} in {entity}/{project}.",
                    stacklevel=2,
                )

    if not results:
        msg = (
            f"No OOD detection results found for method "
            f"'{method_entry.name}' on {dataset}/{base_model} in "
            f"{entity}/{project}. Check that ood_detection.py has been "
            f"run and that the filters match existing runs."
        )
        raise RuntimeError(msg)

    return results
