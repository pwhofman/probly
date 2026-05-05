"""Local cache for wandb run data used by paper plots.

Each cache entry is a single JSON file holding the raw subset of summary
keys we care about (those starting with ``sp/`` or ``ood/``), the run config,
and minimal run metadata. Files are keyed by the same fields used to filter
wandb runs so cache lookups can match without round-tripping the API.

Cache modes:

- ``read`` (default): use cache; fetch missing seeds from wandb and persist
  them back into the cache. Fetched-and-saved seeds are returned alongside
  the cached ones.
- ``refresh``: ignore cache, refetch every run, overwrite the cache files.
- ``off``: bypass the cache entirely (legacy direct-wandb behavior).
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import wandb

from probly_benchmark.paths import CACHE_PATH

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from omegaconf import DictConfig


CacheMode = str  # "read" | "refresh" | "off"

_RELEVANT_PREFIXES = ("sp/", "ood/")


def _summary_key_relevant(key: str) -> bool:
    """Return whether a summary key should be persisted to the cache."""
    return any(key.startswith(p) for p in _RELEVANT_PREFIXES)


def _calibration_segment(calibration: str | None) -> str:
    """Return a safe path segment for the calibration component.

    Args:
        calibration: Calibration name, or ``None`` if no calibration is used.

    Returns:
        Always-present directory segment: ``cal_none`` or ``cal_<name>``.
    """
    return f"cal_{calibration}" if calibration else "cal_none"


def cache_dir(
    entity: str,
    project: str,
    dataset: str,
    base_model: str,
    method: str,
    calibration: str | None = None,
) -> Path:
    """Return the directory holding seed-level cache files for a method.

    Args:
        entity: W&B entity (username or team name).
        project: W&B project name.
        dataset: ID dataset name (e.g. ``"cifar10"``).
        base_model: Base model name (e.g. ``"resnet18"``).
        method: Method name as stored in the run config (``config.method.name``).
        calibration: Optional calibration name (``config.calibration.name``).

    Returns:
        Absolute path to the directory holding ``seed_*.json`` files for this
        ``(method, calibration)`` combination.
    """
    return CACHE_PATH / entity / project / dataset / base_model / method / _calibration_segment(calibration)


def _seed_path(directory: Path, seed: Any) -> Path:  # noqa: ANN401
    """Return the cache file path for a given seed."""
    return directory / f"seed_{seed}.json"


def _json_default(obj: Any) -> Any:  # noqa: ANN401
    """JSON encoder fallback that handles numpy and wandb dict-likes."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    # SummarySubDict and similar dict-likes that aren't dict subclasses.
    if hasattr(obj, "items"):
        return dict(obj.items())
    msg = f"Cannot JSON-serialize value of type {type(obj).__name__}: {obj!r}"
    raise TypeError(msg)


def save_run(record: dict[str, Any]) -> None:
    """Write one run record to the cache.

    The destination path is derived from the record's ``config`` fields.
    Existing files for the same ``(method, calibration, seed)`` are
    overwritten. Numpy arrays are written as plain JSON lists.

    Args:
        record: Run record dict. Must contain ``config`` keys ``dataset``,
            ``base_model``, ``method`` (or ``method.name``), and ``seed``.
            Optional ``calibration`` (or ``calibration.name``) is honored.
            Must also contain ``entity`` and ``project`` at the top level.
    """
    config = record["config"]
    method = config.get("method")
    if isinstance(method, dict):
        method = method.get("name")
    calibration = config.get("calibration")
    if isinstance(calibration, dict):
        calibration = calibration.get("name")

    directory = cache_dir(
        record["entity"],
        record["project"],
        config["dataset"],
        config["base_model"],
        method,
        calibration,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = _seed_path(directory, config["seed"])
    with path.open("w") as fh:
        json.dump(record, fh, default=_json_default)


def load_runs(
    entity: str,
    project: str,
    dataset: str,
    base_model: str,
    method: str,
    calibration: str | None = None,
    seeds: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Load all cached run records for one method.

    Summary list values are coerced back to ``np.ndarray`` on load (mirroring
    what wandb returns), so consumers don't see lists vs arrays depending on
    cache hit/miss.

    Args:
        entity: W&B entity.
        project: W&B project.
        dataset: ID dataset name.
        base_model: Base model name.
        method: Method name.
        calibration: Calibration name, or ``None`` for runs without calibration.
        seeds: If given, restrict to these seeds. ``None`` returns every cached
            seed.

    Returns:
        List of run record dicts in the on-disk schema, sorted by seed.
    """
    directory = cache_dir(entity, project, dataset, base_model, method, calibration)
    if not directory.exists():
        return []

    seed_set: set[Any] | None
    seed_set = set(seeds) if seeds is not None else None

    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("seed_*.json")):
        with path.open() as fh:
            record = json.load(fh)
        summary = record.get("summary", {})
        for key, value in list(summary.items()):
            summary[key] = _coerce_value(value)
        seed = record.get("config", {}).get("seed")
        if seed_set is not None and seed not in seed_set:
            continue
        records.append(record)
    records.sort(key=lambda r: str(r.get("config", {}).get("seed")))
    return records


def _coerce_value(value: Any) -> Any:  # noqa: ANN401
    """Convert wandb summary values into cache-friendly Python objects."""
    if isinstance(value, list):
        try:
            return np.array(value, dtype=float)
        except (ValueError, TypeError):
            return value
    return value


def _is_large_array_placeholder(value: Any) -> bool:  # noqa: ANN401
    """Return ``True`` for the wandb ``{"_type": "large-array", ...}`` placeholder.

    Wandb's :class:`SummarySubDict` is dict-like but not a :class:`dict`
    subclass, so we use duck typing on ``.get`` rather than ``isinstance``.
    """
    getter = getattr(value, "get", None)
    if getter is None:
        return False
    try:
        return getter("_type") == "large-array"
    except TypeError:
        return False


def _summary_dict_from_run(run: wandb.apis.public.Run) -> dict[str, Any]:
    """Extract the relevant summary slice for one wandb run.

    For "large" array values, wandb's public API returns a placeholder dict
    (``{"_type": "large-array", "value": []}``); the real array lives in the
    run's ``wandb-summary.json``. We pull that file once per run if any
    matching key is a placeholder, then resolve every placeholder against it.

    Args:
        run: A wandb public-API Run.

    Returns:
        Mapping of relevant summary keys to fully-materialized values
        (ndarray for arrays, native scalars otherwise).
    """
    matching_keys = [k for k in run.summary.keys() if _summary_key_relevant(k)]  # noqa: SIM118
    fetched: dict[str, Any] = {k: run.summary.get(k) for k in matching_keys}

    needs_full = any(_is_large_array_placeholder(v) for v in fetched.values())
    full_summary: dict[str, Any] | None = None
    if needs_full:
        with tempfile.TemporaryDirectory() as tmpdir:
            run.file("wandb-summary.json").download(replace=True, root=tmpdir)
            with Path(tmpdir, "wandb-summary.json").open() as fh:
                full_summary = json.load(fh)

    summary: dict[str, Any] = {}
    unresolved: list[str] = []
    for key, raw_value in fetched.items():
        resolved: Any = raw_value
        if _is_large_array_placeholder(resolved) and full_summary is not None:
            from_full = full_summary.get(key)
            resolved = from_full if not _is_large_array_placeholder(from_full) else None
        if resolved is None or _is_large_array_placeholder(resolved):
            unresolved.append(key)
            continue
        summary[key] = _coerce_value(resolved)
    if unresolved:
        warnings.warn(
            f"Run {run.id} ({run.name}): {len(unresolved)} large-array key(s) "
            f"unresolved (likely score arrays not uploaded by training): {unresolved}.",
            stacklevel=2,
        )
    return summary


def _config_to_dict(run_config: Any) -> dict[str, Any]:  # noqa: ANN401
    """Best-effort JSON-safe dict view of a wandb run config.

    The wandb config dict can carry SubDicts and back-references to client
    objects that aren't pickle-friendly. Round-tripping through JSON (with
    ``str`` as the fallback for non-serializable objects) drops anything
    that would later break ``pickle.dump``.
    """
    if isinstance(run_config, dict):
        raw = dict(run_config)
    else:
        try:
            raw = dict(run_config.items())
        except AttributeError:
            raw = dict(run_config)
    return json.loads(json.dumps(raw, default=str))


def _measure_decomp_pairs_from_summary(summary: dict[str, Any]) -> set[tuple[str, str]]:
    """Discover (measure, decomposition) pairs present in OOD summary keys.

    Looks at scalar metric keys ``ood/{ood_ds}/{measure}/{decomp}/{metric}``
    and the shared ``ood/{measure}/{decomp}/id_scores`` form, so we can decide
    which (measure, decomposition) artifact name combinations to attempt.
    """
    pairs: set[tuple[str, str]] = set()
    for key in summary:
        if not key.startswith("ood/"):
            continue
        parts = key.split("/")
        if len(parts) == 5:  # ood/{ood_ds}/{measure}/{decomp}/{metric}
            pairs.add((parts[2], parts[3]))
        elif len(parts) == 4 and parts[3] == "id_scores":  # ood/{measure}/{decomp}/id_scores
            pairs.add((parts[1], parts[2]))
    return pairs


def _ood_datasets_from_summary(summary: dict[str, Any], measure: str, decomposition: str) -> set[str]:
    """Discover OOD dataset names with any scalar metric under (measure, decomp)."""
    needle = f"/{measure}/{decomposition}/"
    out: set[str] = set()
    for key in summary:
        if not key.startswith("ood/") or needle not in key:
            continue
        middle = key[len("ood/") : key.index(needle)]
        if "/" not in middle and middle:
            out.add(middle)
    return out


def _try_fetch_artifact_array(
    api: wandb.Api,
    entity: str,
    project: str,
    artifact_name: str,
    npy_filename: str,
) -> np.ndarray | None:
    """Attempt to download an artifact and return its single-array payload.

    Returns ``None`` on miss. Suppresses every wandb / network error so a
    missing artifact never breaks the broader cache fetch.
    """
    qualname = f"{entity}/{project}/{artifact_name}:latest"
    try:
        art = api.artifact(qualname)
        with tempfile.TemporaryDirectory() as td:
            art.download(root=td)
            return np.asarray(np.load(Path(td) / npy_filename))
    except (wandb.errors.CommError, FileNotFoundError, OSError, ValueError):
        return None


def _populate_score_arrays_from_artifacts(
    api: wandb.Api,
    entity: str,
    project: str,
    *,
    method: str,
    dataset: str,
    seed: Any,  # noqa: ANN401
    summary: dict[str, Any],
) -> None:
    """Fill missing score-array keys in ``summary`` from wandb artifacts.

    For every ``(measure, decomposition)`` discovered in the summary scalar
    metrics, attempts to fetch the matching ``id_scores-...`` artifact and a
    per-OOD-dataset ``ood_scores-...`` artifact. Existing inline arrays are
    left untouched.
    """
    pairs = _measure_decomp_pairs_from_summary(summary)
    for measure, decomposition in pairs:
        id_key = f"ood/{measure}/{decomposition}/id_scores"
        if id_key not in summary or _is_large_array_placeholder(summary.get(id_key)):
            arr = _try_fetch_artifact_array(
                api,
                entity,
                project,
                f"id_scores-{method}-{dataset}-{measure}-{decomposition}-seed{seed}",
                "id_scores.npy",
            )
            if arr is not None:
                summary[id_key] = arr

        for ood_ds in _ood_datasets_from_summary(summary, measure, decomposition):
            ood_key = f"ood/{ood_ds}/{measure}/{decomposition}/ood_scores"
            if ood_key in summary and not _is_large_array_placeholder(summary.get(ood_key)):
                continue
            arr = _try_fetch_artifact_array(
                api,
                entity,
                project,
                f"ood_scores-{method}-{dataset}-{ood_ds}-{measure}-{decomposition}-seed{seed}",
                "ood_scores.npy",
            )
            if arr is not None:
                summary[ood_key] = arr


def _build_record_from_run(
    run: wandb.apis.public.Run,
    entity: str,
    project: str,
    api: wandb.Api,
) -> dict[str, Any]:
    """Build a cache-shaped record from one wandb run.

    Score arrays produced by the new artifact-based logging path
    (``id_scores-...`` and ``ood_scores-...`` artifacts) are merged into the
    record's ``summary`` so downstream consumers see one consistent shape.
    """
    config = _config_to_dict(run.config)
    summary = _summary_dict_from_run(run)
    method_cfg = config.get("method")
    method_name = method_cfg.get("name") if isinstance(method_cfg, dict) else None
    dataset = config.get("dataset")
    seed = config.get("seed")
    if isinstance(method_name, str) and isinstance(dataset, str) and seed is not None:
        _populate_score_arrays_from_artifacts(
            api,
            entity,
            project,
            method=method_name,
            dataset=dataset,
            seed=seed,
            summary=summary,
        )
    return {
        "run_id": run.id,
        "name": run.name,
        "entity": entity,
        "project": project,
        "fetched_at": datetime.now(UTC).isoformat(),
        "config": config,
        "summary": summary,
    }


def _fetch_records_from_wandb(
    entity: str,
    project: str,
    dataset: str,
    base_model: str,
    method: str,
    calibration: str | None,
    seeds: Iterable[Any] | None,
) -> list[dict[str, Any]]:
    """Fetch cache-shaped run records from wandb for one method.

    The first finished run encountered per seed wins (matching the existing
    dedup behavior in :mod:`probly_benchmark.plot.utils`).

    Args:
        entity: W&B entity.
        project: W&B project.
        dataset: ID dataset name.
        base_model: Base model name.
        method: Method name.
        calibration: Calibration name, or ``None``.
        seeds: Seeds to filter on; ``None`` for every available seed.

    Returns:
        Run records for the matched runs.
    """
    api = wandb.Api(timeout=60)
    filters: dict[str, Any] = {
        "config.method.name": method,
        "config.dataset": dataset,
        "config.base_model": base_model,
        "state": "finished",
    }
    if calibration:
        filters["config.calibration.name"] = calibration
    if seeds is not None:
        filters["config.seed"] = {"$in": list(seeds)}

    runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at")

    seen_seeds: set[Any] = set()
    records: list[dict[str, Any]] = []
    for run in runs:
        seed = run.config.get("seed")
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        records.append(_build_record_from_run(run, entity, project, api))
    return records


def _method_filters(method_entry: DictConfig | dict[str, Any]) -> tuple[str, str | None]:
    """Pull method name and calibration name out of a config method entry."""
    name = method_entry.get("name")
    calibration = method_entry.get("calibration")
    if not isinstance(name, str):
        msg = f"Method entry must have a string 'name' field; got {method_entry!r}."
        raise TypeError(msg)
    return name, calibration


def fetch_with_cache(
    entity: str,
    project: str,
    *,
    dataset: str,
    base_model: str,
    method_entry: DictConfig | dict[str, Any],
    default_seeds: Iterable[Any] | None,
    mode: CacheMode = "read",
) -> list[dict[str, Any]]:
    """Return cache-shaped run records for one method, using the cache.

    In ``read`` mode, cached records are used as-is and any missing seeds
    are fetched from wandb and saved back to the cache. ``refresh`` ignores
    the cache and refetches, overwriting existing files. ``off`` skips the
    cache entirely.

    Args:
        entity: W&B entity.
        project: W&B project.
        dataset: ID dataset name.
        base_model: Base model name.
        method_entry: One element from ``cfg.methods``. Must have ``name``;
            optionally ``calibration`` and ``seeds``.
        default_seeds: Fallback seed list when ``method_entry`` has no
            ``seeds``. ``None`` means every available seed.
        mode: Cache mode (see module docstring).

    Returns:
        List of run record dicts.
    """
    method, calibration = _method_filters(method_entry)
    entry_seeds = None
    if hasattr(method_entry, "get"):
        entry_seeds = method_entry.get("seeds")
    elif "seeds" in method_entry:
        entry_seeds = method_entry["seeds"]
    seeds = (
        list(entry_seeds) if entry_seeds is not None else (list(default_seeds) if default_seeds is not None else None)
    )

    if mode == "off":
        return _fetch_records_from_wandb(entity, project, dataset, base_model, method, calibration, seeds)

    if mode == "refresh":
        records = _fetch_records_from_wandb(entity, project, dataset, base_model, method, calibration, seeds)
        for record in records:
            save_run(record)
        return records

    if mode != "read":
        msg = f"Unknown cache mode {mode!r}. Expected one of 'read', 'refresh', 'off'."
        raise ValueError(msg)

    cached = load_runs(entity, project, dataset, base_model, method, calibration, seeds)

    if seeds is None:
        if cached:
            return cached
        records = _fetch_records_from_wandb(entity, project, dataset, base_model, method, calibration, None)
        for record in records:
            save_run(record)
        return records

    cached_seeds = {r["config"].get("seed") for r in cached}
    missing = [s for s in seeds if s not in cached_seeds]
    if not missing:
        return cached

    new_records = _fetch_records_from_wandb(entity, project, dataset, base_model, method, calibration, missing)
    for record in new_records:
        save_run(record)
    fetched_seeds = {r["config"].get("seed") for r in new_records}
    still_missing = [s for s in missing if s not in fetched_seeds]
    if still_missing:
        warnings.warn(
            f"No finished wandb run found for method '{method}' on "
            f"{dataset}/{base_model} in {entity}/{project} for seeds={still_missing}.",
            stacklevel=2,
        )
    return [*cached, *new_records]


def select_summary_keys(record: dict[str, Any], predicate: Callable[[str], bool]) -> dict[str, Any]:
    """Filter a record's summary dict by a key predicate.

    Args:
        record: A cache-shaped run record.
        predicate: Callable returning ``True`` for keys to keep.

    Returns:
        Filtered ``{key: value}`` mapping.
    """
    return {k: v for k, v in record.get("summary", {}).items() if predicate(k)}
