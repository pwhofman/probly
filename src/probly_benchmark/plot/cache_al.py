"""Local cache for active-learning wandb run data.

Mirrors the structure of :mod:`probly_benchmark.plot.cache` but tailored to
the AL run shape: per-iteration ``test_accuracy`` lives in run history (not
summary), and the cache key picks up the AL strategy as an extra axis.

Cache layout::

    probly_cache/{entity}/{project}/{dataset_key}/{method}/{strategy}/
        cal_{calibration}/seed_{seed}.json

``dataset_key`` is the value used to identify a dataset in :data:`AL_DATASETS`,
e.g. ``openml_6``, ``openml_155``, ``openml_156``, ``cifar10``,
``fashion_mnist``.

Each JSON record:

.. code-block:: json

    {
      "run_id": "...",
      "name": "al_dropout_openml_xh5xzvg4",
      "entity": "probly",
      "project": "al_openml_v1600_0505",
      "fetched_at": "2026-...",
      "config": {seed, dataset.{name,openml_id}, method.name, al_strategy.name, calibration.name, ...},
      "summary": {"nauc": 0.72, "final_accuracy": 0.70},
      "history": [{"iteration": 0, "labeled_size": 100, "test_accuracy": 0.755}, ...]
    }

Cache modes (``read | refresh | off``) match :mod:`probly_benchmark.plot.cache`.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import wandb

from probly_benchmark.paths import CACHE_PATH

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from omegaconf import DictConfig


CacheMode = str  # "read" | "refresh" | "off"


# Summary scalars we care about. Anything else logged on the run is ignored.
_SUMMARY_KEYS = ("nauc", "final_accuracy")

# History columns we keep. Wandb adds a few internals (``_step``, ``_runtime``,
# ``_timestamp``); only the AL signals matter for plotting.
_HISTORY_KEYS = ("iteration", "labeled_size", "test_accuracy")


def _calibration_segment(calibration: str | None) -> str:
    return f"cal_{calibration}" if calibration else "cal_none"


def dataset_key(name: str, openml_id: int | str | None) -> str:
    """Build the path-safe dataset key from a config's dataset block.

    ``openml`` datasets need the integer id as part of the key so different
    OpenML tasks don't collide on disk; everything else uses ``name`` directly.
    """
    if name == "openml":
        if openml_id is None:
            msg = "openml dataset entry must include an 'openml_id'."
            raise ValueError(msg)
        return f"openml_{openml_id}"
    return name


def cache_dir(
    entity: str,
    project: str,
    ds_key: str,
    method: str,
    strategy: str,
    calibration: str | None = None,
) -> Path:
    """Return the directory holding seed-level AL cache files for one combo."""
    return CACHE_PATH / entity / project / ds_key / method / strategy / _calibration_segment(calibration)


def _seed_path(directory: Path, seed: Any) -> Path:  # noqa: ANN401
    return directory / f"seed_{seed}.json"


def _json_default(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "items"):
        return dict(obj.items())
    msg = f"Cannot JSON-serialize value of type {type(obj).__name__}: {obj!r}"
    raise TypeError(msg)


def save_run(record: dict[str, Any]) -> None:
    """Write one AL run record to the cache.

    The destination is derived from the record's ``config`` fields.
    """
    config = record["config"]
    method = config.get("method")
    if isinstance(method, dict):
        method = method.get("name")
    strategy = config.get("al_strategy")
    if isinstance(strategy, dict):
        strategy = strategy.get("name")
    calibration = config.get("calibration")
    if isinstance(calibration, dict):
        calibration = calibration.get("name")
    ds = config.get("dataset", {})
    ds_name = ds.get("name") if isinstance(ds, dict) else ds
    ds_id = ds.get("openml_id") if isinstance(ds, dict) else None

    directory = cache_dir(
        record["entity"],
        record["project"],
        dataset_key(ds_name, ds_id),
        method,
        strategy,
        calibration,
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = _seed_path(directory, config["seed"])
    with path.open("w") as fh:
        json.dump(record, fh, default=_json_default)


def load_runs(
    entity: str,
    project: str,
    ds_key: str,
    method: str,
    strategy: str,
    calibration: str | None = None,
    seeds: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Load cached AL run records for one ``(method, strategy, calibration)`` combo."""
    directory = cache_dir(entity, project, ds_key, method, strategy, calibration)
    if not directory.exists():
        return []

    seed_set: set[Any] | None
    seed_set = set(seeds) if seeds is not None else None

    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("seed_*.json")):
        with path.open() as fh:
            record = json.load(fh)
        seed = record.get("config", {}).get("seed")
        if seed_set is not None and seed not in seed_set:
            continue
        records.append(record)
    records.sort(key=lambda r: str(r.get("config", {}).get("seed")))
    return records


def _config_to_dict(run_config: Any) -> dict[str, Any]:  # noqa: ANN401
    """JSON-safe dict view of a wandb run config (drops non-serializable refs)."""
    if isinstance(run_config, dict):
        raw = dict(run_config)
    else:
        try:
            raw = dict(run_config.items())
        except AttributeError:
            raw = dict(run_config)
    return json.loads(json.dumps(raw, default=str))


def _summary_from_run(run: wandb.apis.public.Run) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _SUMMARY_KEYS:
        v = run.summary.get(key)
        if v is None:
            continue
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            out[key] = v
    return out


def _history_from_run(run: wandb.apis.public.Run) -> list[dict[str, Any]]:
    """Pull per-iteration history rows for the AL signals we care about."""
    df = run.history(samples=1000, pandas=True, keys=list(_HISTORY_KEYS))
    if df is None or len(df) == 0:
        return []
    columns = set(df.columns)  # ty: ignore[unresolved-attribute]
    int_keys = {"iteration", "labeled_size"}
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():  # ty: ignore[unresolved-attribute]
        item: dict[str, Any] = {}
        for key in _HISTORY_KEYS:
            if key not in columns:
                continue
            value = row[key]
            if value is None:
                continue
            try:
                item[key] = int(value) if key in int_keys else float(value)
            except (TypeError, ValueError):
                continue
        if "iteration" in item:
            rows.append(item)
    rows.sort(key=lambda r: r.get("iteration", 0))
    return rows


def _build_record_from_run(run: wandb.apis.public.Run, entity: str, project: str) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "name": run.name,
        "entity": entity,
        "project": project,
        "fetched_at": datetime.now(UTC).isoformat(),
        "config": _config_to_dict(run.config),
        "summary": _summary_from_run(run),
        "history": _history_from_run(run),
    }


def _fetch_records_from_wandb(
    entity: str,
    project: str,
    ds_key: str,
    method: str,
    strategy: str,
    calibration: str | None,
    seeds: Iterable[Any] | None,
) -> list[dict[str, Any]]:
    api = wandb.Api(timeout=60)

    filters: dict[str, Any] = {
        "config.method.name": method,
        "config.al_strategy.name": strategy,
        "state": "finished",
    }
    if ds_key.startswith("openml_"):
        filters["config.dataset.name"] = "openml"
        filters["config.dataset.openml_id"] = int(ds_key.removeprefix("openml_"))
    else:
        filters["config.dataset.name"] = ds_key
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
        records.append(_build_record_from_run(run, entity, project))
    return records


def _method_filters(method_entry: DictConfig | dict[str, Any]) -> tuple[str, str | None]:
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
    ds_key: str,
    method_entry: DictConfig | dict[str, Any],
    strategy: str,
    default_seeds: Iterable[Any] | None,
    mode: CacheMode = "read",
) -> list[dict[str, Any]]:
    """Return cache-shaped AL records for one (method, strategy, dataset) combo.

    Args:
        entity: W&B entity.
        project: W&B project (e.g. ``"al_openml_v1600_0505"``).
        ds_key: ``dataset_key(...)`` value (e.g. ``"openml_6"``).
        method_entry: Method config entry. Must have ``name``; optionally
            ``calibration`` and ``seeds``.
        strategy: Acquisition strategy (``"uncertainty"``, ``"random"``, ...).
        default_seeds: Fallback seed list when ``method_entry`` has no seeds.
        mode: ``"read"`` | ``"refresh"`` | ``"off"``.

    Returns:
        List of run-record dicts.
    """
    method, calibration = _method_filters(method_entry)
    entry_seeds = method_entry.get("seeds") if hasattr(method_entry, "get") else None
    seeds = (
        list(entry_seeds) if entry_seeds is not None else (list(default_seeds) if default_seeds is not None else None)
    )

    if mode == "off":
        return _fetch_records_from_wandb(entity, project, ds_key, method, strategy, calibration, seeds)

    if mode == "refresh":
        records = _fetch_records_from_wandb(entity, project, ds_key, method, strategy, calibration, seeds)
        for record in records:
            save_run(record)
        return records

    if mode != "read":
        msg = f"Unknown cache mode {mode!r}. Expected one of 'read', 'refresh', 'off'."
        raise ValueError(msg)

    cached = load_runs(entity, project, ds_key, method, strategy, calibration, seeds)

    if seeds is None:
        if cached:
            return cached
        records = _fetch_records_from_wandb(entity, project, ds_key, method, strategy, calibration, None)
        for record in records:
            save_run(record)
        return records

    cached_seeds = {r["config"].get("seed") for r in cached}
    missing = [s for s in seeds if s not in cached_seeds]
    if not missing:
        return cached

    new_records = _fetch_records_from_wandb(entity, project, ds_key, method, strategy, calibration, missing)
    for record in new_records:
        save_run(record)
    fetched_seeds = {r["config"].get("seed") for r in new_records}
    still_missing = [s for s in missing if s not in fetched_seeds]
    if still_missing:
        warnings.warn(
            f"No finished AL run found for method '{method}' / strategy '{strategy}' on "
            f"{ds_key} in {entity}/{project} for seeds={still_missing}.",
            stacklevel=2,
        )
    return [*cached, *new_records]
