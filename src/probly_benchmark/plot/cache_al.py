"""Local cache for active-learning wandb run data.

Two-stage flow:

1. **Bulk download** (``cache_bulk_al.py``) -- pulls every finished run from
   each source project verbatim into a raw store at
   ``probly_cache/_raw/{entity}/{source_project}/{run_id}.json``. No
   filtering, no dedup; just (config, summary, history) per run.
2. **Analyze + dedup** (``cache_analyze_al.py``) -- walks the raw store,
   applies the *complete* predicate (state, summary, history length),
   classifies each run by the cache combo
   ``(ds_key, method, strategy, notion, loss, cal, seed)``, picks the
   newest complete run per combo, and writes the merged sink at
   ``probly_cache/{entity}/{sink}/{ds_key}/{method}/{strategy}/notion_{notion}/loss_{loss}/cal_{cal}/seed_{seed}.json``.

For non-uncertainty strategies the ``notion`` segment is ``notion_n_a`` (the
strategy ignores the notion field).

Each merged record carries ``source_project`` + ``source_run_id`` so the
provenance of every cache cell stays traceable to the wandb run that
produced it.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from probly_benchmark.paths import CACHE_PATH

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import wandb


# Summary scalars we care about. Anything else logged on the run is ignored.
_SUMMARY_KEYS = ("nauc", "final_accuracy")

# History columns we keep. Wandb adds a few internals (``_step``, ``_runtime``,
# ``_timestamp``); only the AL signals matter for plotting.
_HISTORY_KEYS = ("iteration", "labeled_size", "test_accuracy")


_DEFAULT_NOTION = "epistemic"
_DEFAULT_LOSS = "cross_entropy"
_DEFAULT_CAL = "none"
_NOTION_NA = "n_a"


def _calibration_segment(calibration: str | None) -> str:
    return f"cal_{calibration or _DEFAULT_CAL}"


def _notion_segment(notion: str | None) -> str:
    return f"notion_{notion or _DEFAULT_NOTION}"


def _loss_segment(supervised_loss: str | None) -> str:
    return f"loss_{supervised_loss or _DEFAULT_LOSS}"


def effective_notion(strategy: str, notion: str | None) -> str:
    """Notion only matters for the ``uncertainty`` strategy; collapse otherwise."""
    if strategy == "uncertainty":
        return notion or _DEFAULT_NOTION
    return _NOTION_NA


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
    notion: str | None = None,
    supervised_loss: str | None = None,
    calibration: str | None = None,
) -> Path:
    """Return the directory holding seed-level AL cache files for one combo."""
    return (
        CACHE_PATH
        / entity
        / project
        / ds_key
        / method
        / strategy
        / _notion_segment(effective_notion(strategy, notion))
        / _loss_segment(supervised_loss)
        / _calibration_segment(calibration)
    )


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


def _config_axes(config: dict[str, Any]) -> tuple[str, str, str | None, str | None, str | None]:
    """Pull (method, strategy, notion, supervised_loss, calibration) out of a record's config."""
    method_v = config.get("method")
    method = method_v.get("name") if isinstance(method_v, dict) else method_v
    strat_v = config.get("al_strategy")
    if isinstance(strat_v, dict):
        strategy = strat_v.get("name")
        notion = strat_v.get("notion")
    else:
        strategy = strat_v
        notion = None
    cal_v = config.get("calibration")
    calibration = cal_v.get("name") if isinstance(cal_v, dict) else cal_v
    loss_v = config.get("supervised_loss")
    loss = loss_v.get("name") if isinstance(loss_v, dict) else loss_v
    return str(method), str(strategy), notion, loss, calibration


def save_run(record: dict[str, Any]) -> None:
    """Write one AL run record to the cache.

    The destination is derived from the record's ``config`` fields.
    """
    config = record["config"]
    method, strategy, notion, loss, calibration = _config_axes(config)
    ds = config.get("dataset", {})
    ds_name = ds.get("name") if isinstance(ds, dict) else ds
    ds_id = ds.get("openml_id") if isinstance(ds, dict) else None

    directory = cache_dir(
        record["entity"],
        record["project"],
        dataset_key(ds_name, ds_id),
        method,
        strategy,
        notion=notion,
        supervised_loss=loss,
        calibration=calibration,
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
    notion: str | None = None,
    supervised_loss: str | None = None,
    calibration: str | None = None,
    seeds: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Load cached AL run records for one ``(method, strategy, ...)`` combo."""
    directory = cache_dir(
        entity,
        project,
        ds_key,
        method,
        strategy,
        notion=notion,
        supervised_loss=supervised_loss,
        calibration=calibration,
    )
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


# --------------------------------------------------------------------------
# Bulk download primitives (used by ``cache_bulk_al.py``)
# --------------------------------------------------------------------------


def raw_dir(entity: str, source_project: str) -> Path:
    """Return the raw-cache directory for ``{entity}/{source_project}``."""
    return CACHE_PATH / "_raw" / entity / source_project


def raw_record_path(entity: str, source_project: str, run_id: str) -> Path:
    """Return the path to a single raw cache file."""
    return raw_dir(entity, source_project) / f"{run_id}.json"


def save_raw_record(record: dict[str, Any]) -> None:
    """Write one raw run record to the bulk store."""
    path = raw_record_path(record["entity"], record["source_project"], record["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(record, fh, default=_json_default)


def iter_raw_records(entity: str, source_project: str) -> Iterable[dict[str, Any]]:
    """Yield every raw record cached for ``{entity}/{source_project}``."""
    directory = raw_dir(entity, source_project)
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        with path.open() as fh:
            yield json.load(fh)


def build_raw_record(
    run: wandb.apis.public.Run,
    entity: str,
    source_project: str,
) -> dict[str, Any]:
    """Build a raw record from a wandb run. Always succeeds (no filtering)."""
    return {
        "run_id": run.id,
        "name": run.name,
        "entity": entity,
        "source_project": source_project,
        "state": run.state,
        "created_at": str(run.created_at),
        "fetched_at": datetime.now(UTC).isoformat(),
        "config": _config_to_dict(run.config),
        "summary": _summary_from_run(run),
        "history": _history_from_run(run),
    }


# --------------------------------------------------------------------------
# Classification (used by ``cache_analyze_al.py``)
# --------------------------------------------------------------------------


# Status keys returned by :func:`classify_raw_record`.
STATUS_COMPLETE = "complete"
STATUS_WRONG_STATE = "wrong_state"
STATUS_MISSING_SUMMARY = "missing_summary"
STATUS_MISSING_N_ITER = "missing_n_iterations"
STATUS_TRUNCATED = "truncated_history"
STATUS_MISSING_AXES = "missing_combo_axes"


def _check_summary(summary: dict[str, Any]) -> bool:
    """Return ``True`` iff ``nauc`` and ``final_accuracy`` are present and parseable."""
    nauc = summary.get("nauc")
    final_acc = summary.get("final_accuracy")
    if nauc is None or final_acc is None:
        return False
    try:
        float(nauc)
        float(final_acc)
    except (TypeError, ValueError):
        return False
    return True


def _extract_combo(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the cache-cell axes out of a config; return ``None`` if any are missing."""
    method, strategy, notion, loss, calibration = _config_axes(cfg)
    if method == "None" or strategy == "None":
        return None

    ds = cfg.get("dataset") or {}
    if isinstance(ds, dict):
        ds_name = ds.get("name")
        ds_id = ds.get("openml_id")
    else:
        ds_name = ds
        ds_id = None
    if not isinstance(ds_name, str):
        return None
    try:
        ds_key = dataset_key(ds_name, ds_id)
    except (ValueError, KeyError):
        return None

    seed = cfg.get("seed")
    if seed is None:
        return None

    return {
        "ds_key": ds_key,
        "method": method,
        "strategy": strategy,
        "notion": notion,
        "supervised_loss": loss,
        "calibration": calibration,
        "seed": seed,
    }


def classify_raw_record(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Return ``(status, combo)`` for one raw record.

    ``combo`` is a dict ``{ds_key, method, strategy, notion, supervised_loss,
    calibration, seed}`` when ``status == "complete"``, else ``None``. ``status``
    is one of the ``STATUS_*`` constants and tells you why a record was rejected.
    """
    if record.get("state") != "finished":
        return STATUS_WRONG_STATE, None

    if not _check_summary(record.get("summary") or {}):
        return STATUS_MISSING_SUMMARY, None

    cfg = record.get("config") or {}
    n_iter_raw = cfg.get("n_iterations")
    try:
        n_iter = int(n_iter_raw) if n_iter_raw is not None else None
    except (TypeError, ValueError):
        n_iter = None
    if n_iter is None:
        return STATUS_MISSING_N_ITER, None

    history = record.get("history") or []
    if len(history) < n_iter + 1:
        return STATUS_TRUNCATED, None

    combo = _extract_combo(cfg)
    if combo is None:
        return STATUS_MISSING_AXES, None
    return STATUS_COMPLETE, combo


def to_sink_record(raw_record: dict[str, Any], sink_project: str) -> dict[str, Any]:
    """Adapt a raw record into the merged-sink record shape consumed by ``save_run``.

    Strips bulk-store-only fields and rewrites ``project`` to point at the
    sink directory segment so :func:`save_run` lands the file in the merged
    cache.
    """
    sink_record = dict(raw_record)
    sink_record["entity"] = raw_record["entity"]
    sink_record["project"] = sink_project
    sink_record["source_project"] = raw_record["source_project"]
    sink_record["source_run_id"] = raw_record["run_id"]
    return sink_record
