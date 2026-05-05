"""Shared utilities for probly_benchmark plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly_benchmark.paths import FIGURE_PATH
from probly_benchmark.plot import cache

_METHOD_CONFIG_DIR = Path(__file__).parent.parent / "configs" / "method"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_save_path(save_path: str | None) -> Path:
    """Resolve a ``save_path`` config value to an absolute output directory.

    Args:
        save_path: User-supplied save path, or ``None`` to fall back to
            :data:`probly_benchmark.paths.FIGURE_PATH`. Absolute paths are used
            as-is; relative paths are interpreted relative to the repository
            root, which is convenient for landing figures in
            ``paper_artifacts/<topic>/``.

    Returns:
        Absolute :class:`pathlib.Path` for the figure output directory.
    """
    if save_path is None:
        return FIGURE_PATH
    candidate = Path(save_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _REPO_ROOT / candidate


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


def _as_array(value: Any) -> np.ndarray:  # noqa: ANN401
    """Convert a cached or wandb summary value to a float ndarray."""
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    return np.asarray(value, dtype=float)


def fetch_sp_runs(
    entity: str,
    project: str,
    method_entry: DictConfig,
    dataset: str,
    base_model: str,
    default_seeds: list[int] | None = None,
    measure: str = "default",
    decomposition: str = "total",
    cache_mode: cache.CacheMode = "read",
) -> list[dict[str, Any]]:
    """Fetch selective prediction results from wandb (with caching) for one method.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        method_entry: One element from ``cfg.methods``. Must have a ``name`` field.
            Optionally ``calibration``, ``seeds``, ``measure``, and
            ``decomposition`` (override the function-level defaults).
        dataset: Dataset name as stored in the run config (e.g. ``"cifar10"``).
        base_model: Base model name as stored in the run config.
        default_seeds: Seeds to filter on when the method entry has no ``seeds``
            field. ``None`` means all available seeds.
        measure: Uncertainty measure used when logging (e.g. ``"default"``).
        decomposition: Uncertainty decomposition used when logging (e.g. ``"total"``).
        cache_mode: ``"read"`` (default), ``"refresh"``, or ``"off"`` (see
            :mod:`probly_benchmark.plot.cache`).

    Returns:
        List of dicts, one per matching run, each with keys:

        - ``bin_losses``: ``np.ndarray`` of per-bin 0/1 error rates.
        - ``auroc``: Float AUROC of the loss-rejection curve.
        - ``seed``: Integer seed of the run.

    Raises:
        RuntimeError: If no runs with selective prediction results are found.
    """
    records = cache.fetch_with_cache(
        entity,
        project,
        dataset=dataset,
        base_model=base_model,
        method_entry=method_entry,
        default_seeds=default_seeds,
        mode=cache_mode,
    )

    entry_measure = method_entry.get("measure") or measure
    entry_decomp = method_entry.get("decomposition") or decomposition
    prefix = f"sp/{entry_measure}/{entry_decomp}"

    results: list[dict[str, Any]] = []
    for record in records:
        seed = record.get("config", {}).get("seed")
        summary = record.get("summary", {})
        bin_losses = summary.get(f"{prefix}/bin_losses")
        auroc = summary.get(f"{prefix}/auroc")
        if bin_losses is None or auroc is None:
            warnings.warn(
                f"Run {record.get('run_id')} ({record.get('name')}) matched filters "
                "but has no selective prediction results. Was selective_prediction.py "
                "run for this run?",
                stacklevel=2,
            )
            continue
        results.append(
            {
                "bin_losses": _as_array(bin_losses),
                "auroc": float(auroc),
                "seed": seed,
            }
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


def _discover_ood_datasets_from_summary(summary: dict[str, Any], measure: str, decomposition: str) -> list[str]:
    """Return OOD dataset names that have ``ood_scores`` arrays in the summary."""
    suffix = f"/{measure}/{decomposition}/ood_scores"
    datasets: list[str] = []
    for key in summary:
        if key.startswith("ood/") and key.endswith(suffix):
            middle = key[len("ood/") : -len(suffix)]
            if "/" not in middle:
                datasets.append(middle)
    return sorted(datasets)


def _discover_ood_datasets_with_metrics(summary: dict[str, Any], measure: str, decomposition: str) -> list[str]:
    """Discover OOD datasets that have any scalar metric under the prefix.

    Used as a fallback when score arrays are missing (placeholders that wandb
    did not materialize) but scalar metrics like ``auroc`` are present.
    """
    needle = f"/{measure}/{decomposition}/"
    datasets: set[str] = set()
    for key in summary:
        if not key.startswith("ood/") or needle not in key:
            continue
        middle = key[len("ood/") : key.index(needle)]
        if "/" not in middle:
            datasets.add(middle)
    return sorted(datasets)


def _collect_ood_scalar_metrics_from_summary(summary: dict[str, Any], prefix: str) -> dict[str, float]:
    """Collect scalar OOD metric values under a prefix from a cached summary."""
    strip = prefix + "/"
    metrics: dict[str, float] = {}
    for key, value in summary.items():
        if not key.startswith(strip):
            continue
        suffix = key[len(strip) :]
        if suffix in ("ood_scores", "id_scores"):
            continue
        if value is None:
            continue
        try:
            metrics[suffix] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


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
    cache_mode: cache.CacheMode = "read",
) -> dict[str, list[dict[str, Any]]]:
    """Fetch OOD detection results from wandb (with caching) for one method.

    All OOD dataset results live on the same training run as separate summary
    key prefixes ``ood/{ood_dataset}/{measure}/{decomposition}/...``, so a
    single cached record covers every OOD dataset for one ``(method, seed)``.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        method_entry: One element from ``cfg.methods``. Must have a ``name`` field.
            Optionally ``calibration``, ``seeds``, ``measure``, and
            ``decomposition`` (override the function-level defaults).
        dataset: In-distribution dataset name as stored in the run config.
        base_model: Base model name as stored in the run config.
        ood_datasets: OOD dataset names to collect. ``None`` means use every
            dataset present in the cached/fetched runs.
        default_seeds: Seeds to filter on when the method entry has no ``seeds``
            field. ``None`` means all available seeds.
        measure: Uncertainty measure used when logging (e.g. ``"default"``).
        decomposition: Uncertainty decomposition used when logging (e.g.
            ``"epistemic"``).
        cache_mode: ``"read"`` (default), ``"refresh"``, or ``"off"`` (see
            :mod:`probly_benchmark.plot.cache`).

    Returns:
        Dict mapping each OOD dataset name to a list of run-level dicts. Each
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
    records = cache.fetch_with_cache(
        entity,
        project,
        dataset=dataset,
        base_model=base_model,
        method_entry=method_entry,
        default_seeds=default_seeds,
        mode=cache_mode,
    )

    entry_measure = method_entry.get("measure") or measure
    entry_decomp = method_entry.get("decomposition") or decomposition
    id_scores_key = f"ood/{entry_measure}/{entry_decomp}/id_scores"

    empty = np.empty((0,), dtype=float)
    results_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        seed = record.get("config", {}).get("seed")
        summary = record.get("summary", {})
        id_scores_raw = summary.get(id_scores_key)
        id_scores = _as_array(id_scores_raw) if id_scores_raw is not None else empty

        available = (
            ood_datasets
            if ood_datasets is not None
            else _discover_ood_datasets_from_summary(summary, entry_measure, entry_decomp)
        )
        # Fall back to scalar-only OOD datasets discovered from the summary
        # (e.g. only auroc/aupr/fpr were logged because score arrays were
        # placeholders). This keeps bar plots working for those runs.
        if not available:
            available = _discover_ood_datasets_with_metrics(summary, entry_measure, entry_decomp)

        for ood_ds in available:
            ood_prefix = f"ood/{ood_ds}/{entry_measure}/{entry_decomp}"
            ood_scores_raw = summary.get(f"{ood_prefix}/ood_scores")
            metrics = _collect_ood_scalar_metrics_from_summary(summary, ood_prefix)
            auroc_raw = summary.get(f"{ood_prefix}/auroc")
            if ood_scores_raw is None and not metrics and auroc_raw is None:
                continue
            results_by_dataset.setdefault(ood_ds, []).append(
                {
                    "id_scores": id_scores,
                    "ood_scores": _as_array(ood_scores_raw) if ood_scores_raw is not None else empty,
                    "auroc": float(auroc_raw) if auroc_raw is not None else None,
                    "metrics": metrics,
                    "seed": seed,
                }
            )

    if not results_by_dataset:
        msg = (
            f"No OOD detection results found for method '{method_entry.name}' "
            f"on {dataset}/{base_model} in {entity}/{project}. "
            "Check that ood_detection.py has been run and that the filters match existing runs."
        )
        raise RuntimeError(msg)

    return results_by_dataset
