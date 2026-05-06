"""Shared utilities for probly_benchmark plotting."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import warnings

import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb

_METHOD_CONFIG_DIR = Path(__file__).parent.parent / "configs" / "method"


def resolve_label(entry: DictConfig) -> str:
    """Resolve a human-readable label for a method entry.

    Looks up the label in the following order: the ``label`` field of the
    method entry itself, the ``label`` field at the top of the method's config
    file, the ``method.label`` field of that config, and finally the method
    name as a fallback.

    Args:
        entry: One element from ``cfg.methods``. Must have a ``name`` field
            and may have an optional ``label`` field.

    Returns:
        The resolved display label as a string.
    """
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


def _get_summary_value(run: wandb.apis.public.Run, key: str) -> object:
    """Retrieve a summary value, downloading the full JSON for large arrays.

    wandb's public API returns a SummarySubDict placeholder for large arrays
    rather than the actual data. This helper falls back to wandb-summary.json
    so callers always get the real value.

    Args:
        run: A wandb public API Run object.
        key: Summary key to retrieve.

    Returns:
        The summary value, or ``None`` if the key is absent.
    """
    value = run.summary.get(key)
    if value is None:
        return None
    raw = run.summary._json_dict.get(key)  # noqa: SLF001
    if isinstance(raw, dict) and raw.get("_type") == "large-array":
        with tempfile.TemporaryDirectory() as tmpdir:
            run.file("wandb-summary.json").download(replace=True, root=tmpdir)
            with Path(tmpdir, "wandb-summary.json").open() as fh:
                full_summary = json.load(fh)
        return full_summary.get(key)
    return value


def fetch_sp_runs(
    entity: str,
    project: str,
    method_entry: DictConfig,
    dataset: str,
    base_model: str,
    default_seeds: list[int] | None = None,
    measure: str = "default",
    decomposition: str = "total",
) -> list[dict[str, Any]]:
    """Fetch selective prediction results from wandb for one method entry.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        method_entry: One element from ``cfg.methods``. Must have a ``name`` field.
            Optionally ``calibration`` (maps to ``config.calibration.name``),
            ``seeds`` (overrides ``default_seeds``), ``measure``, and
            ``decomposition`` (override the function-level defaults).
        dataset: Dataset name as stored in the run config (e.g. ``"cifar10"``).
        base_model: Base model name as stored in the run config (e.g. ``"resnet18"``).
        default_seeds: Seeds to filter on when the method entry has no ``seeds``
            field. ``None`` means all available seeds.
        measure: Uncertainty measure used when logging (e.g. ``"default"``).
        decomposition: Uncertainty decomposition used when logging (e.g. ``"total"``).

    Returns:
        List of dicts, one per matching run, each with keys:

        - ``bin_losses``: ``np.ndarray`` of per-bin 0/1 error rates.
        - ``auroc``: Float AUROC of the loss-rejection curve.
        - ``seed``: Integer seed of the run.

    Raises:
        RuntimeError: If no runs with selective prediction results are found.
    """
    api = wandb.Api(timeout=60)

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

    entry_measure = method_entry.get("measure") or measure
    entry_decomp = method_entry.get("decomposition") or decomposition
    prefix = f"sp/{entry_measure}/{entry_decomp}"

    seen_seeds: set[Any] = set()
    results = []
    for run in runs:
        seed = run.config.get("seed")
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        bin_losses = run.summary.get(f"{prefix}/bin_losses")
        auroc = run.summary.get(f"{prefix}/auroc")
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


def _collect_ood_scalar_metrics(run: wandb.apis.public.Run, prefix: str) -> dict[str, float]:
    """Collect all scalar summary values under ``prefix`` into a flat dict.

    The ``prefix + "/"`` portion is stripped from the keys, and the score
    arrays (ending in ``/ood_scores`` or ``/id_scores``) are excluded.

    Args:
        run: A wandb public API Run object.
        prefix: The summary key prefix to filter on, e.g.
            ``"ood/cifar100/default/epistemic"``.

    Returns:
        Mapping from metric name (without the prefix) to float value.
    """
    strip = prefix + "/"
    metrics: dict[str, float] = {}
    for key in run.summary.keys():  # noqa: SIM118
        if not key.startswith(strip):
            continue
        suffix = key[len(strip) :]
        if suffix in ("ood_scores", "id_scores"):
            continue
        value = run.summary.get(key)
        if value is None:
            continue
        try:
            metrics[suffix] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _discover_ood_datasets(run: wandb.apis.public.Run, measure: str, decomposition: str) -> list[str]:
    """Return OOD dataset names that have results logged on this run.

    Scans summary keys for the pattern
    ``ood/{ood_dataset}/{measure}/{decomposition}/ood_scores`` and returns
    the unique ``ood_dataset`` values found.

    Args:
        run: A wandb public API Run object.
        measure: Uncertainty measure segment used in the key (e.g. ``"default"``).
        decomposition: Uncertainty decomposition segment (e.g. ``"epistemic"``).

    Returns:
        Sorted list of OOD dataset name strings.
    """
    suffix = f"/{measure}/{decomposition}/ood_scores"
    datasets: list[str] = []
    for key in run.summary.keys():  # noqa: SIM118
        if key.startswith("ood/") and key.endswith(suffix):
            middle = key[len("ood/") : -len(suffix)]
            if "/" not in middle:
                datasets.append(middle)
    return sorted(datasets)


def fetch_ood_runs(
    entity: str,
    project: str,
    method_entry: DictConfig,
    dataset: str,
    base_model: str,
    ood_datasets: list[str] | None = None,
    default_seeds: list[int] | None = None,
    measure: str = "default",
    decomposition: str = "epistemic",
) -> dict[str, list[dict[str, Any]]]:
    """Fetch OOD detection results from wandb for one method entry.

    All OOD dataset results live on the same training run as separate summary
    key prefixes ``ood/{ood_dataset}/{measure}/{decomposition}/...``, so a
    single API query retrieves data for every OOD dataset.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        method_entry: One element from ``cfg.methods``. Must have a ``name`` field.
            Optionally ``calibration`` (maps to ``config.calibration.name``),
            ``seeds`` (overrides ``default_seeds``), ``measure``, and
            ``decomposition`` (override the function-level defaults).
        dataset: In-distribution dataset name as stored in the run config.
        base_model: Base model name as stored in the run config.
        ood_datasets: OOD dataset names to collect. ``None`` means auto-discover
            all datasets that have results logged on the fetched runs.
        default_seeds: Seeds to filter on when the method entry has no ``seeds``
            field. ``None`` means all available seeds.
        measure: Uncertainty measure used when logging (e.g. ``"default"``).
        decomposition: Uncertainty decomposition used when logging (e.g.
            ``"epistemic"``).

    Returns:
        Dict mapping each OOD dataset name to a list of run-level dicts.  Each
        dict has keys:

        - ``id_scores``: ``np.ndarray`` of in-distribution uncertainty scores.
        - ``ood_scores``: ``np.ndarray`` of OOD uncertainty scores.
        - ``auroc``: Float AUROC, or ``None`` if not logged.
        - ``metrics``: ``dict[str, float]`` of scalar summary values under the
          OOD prefix (score arrays excluded), with the prefix stripped.
        - ``seed``: Integer seed of the run.

    Raises:
        RuntimeError: If no runs with OOD results are found for any dataset.
    """
    api = wandb.Api(timeout=60)

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

    entry_measure = method_entry.get("measure") or measure
    entry_decomp = method_entry.get("decomposition") or decomposition
    id_scores_key = f"ood/{entry_measure}/{entry_decomp}/id_scores"

    seen_seeds: set[Any] = set()
    results_by_dataset: dict[str, list[dict[str, Any]]] = {}

    for run in runs:
        seed = run.config.get("seed")
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        id_scores_raw = _get_summary_value(run, id_scores_key)
        if id_scores_raw is None:
            warnings.warn(
                f"Run {run.id} ({run.name}) matched filters but has no OOD id_scores. "
                "Was ood_detection.py run for this run?",
                stacklevel=2,
            )
            continue
        id_scores = np.array(id_scores_raw)

        available = (
            ood_datasets if ood_datasets is not None else _discover_ood_datasets(run, entry_measure, entry_decomp)
        )
        for ood_ds in available:
            ood_prefix = f"ood/{ood_ds}/{entry_measure}/{entry_decomp}"
            ood_scores_raw = _get_summary_value(run, f"{ood_prefix}/ood_scores")
            if ood_scores_raw is None:
                continue
            metrics = _collect_ood_scalar_metrics(run, ood_prefix)
            auroc_key = f"{ood_prefix}/auroc"
            results_by_dataset.setdefault(ood_ds, []).append(
                {
                    "id_scores": id_scores,
                    "ood_scores": np.array(ood_scores_raw),
                    "auroc": float(run.summary[auroc_key]) if run.summary.get(auroc_key) is not None else None,
                    "metrics": metrics,
                    "seed": seed,
                }
            )

    if seeds is not None:
        for ood_ds, ds_results in results_by_dataset.items():
            found_seeds = {r["seed"] for r in ds_results}
            for s in seeds:
                if s not in found_seeds:
                    warnings.warn(
                        f"No finished run found for method '{method_entry.name}' "
                        f"seed={s} on {dataset}/{base_model}/{ood_ds} in {entity}/{project}.",
                        stacklevel=2,
                    )

    if not results_by_dataset:
        msg = (
            f"No OOD detection results found for method '{method_entry.name}' "
            f"on {dataset}/{base_model} in {entity}/{project}. "
            "Check that ood_detection.py has been run and that the filters match existing runs."
        )
        raise RuntimeError(msg)

    return results_by_dataset
