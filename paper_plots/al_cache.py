"""Local cache for active-learning runs (separate from the SP/OOD cache).

Two-stage flow:

1. **Bulk download** (:mod:`paper_plots.fetch_al`) writes every finished run
   from each source project verbatim to::

       paper_plots/.cache_al/_raw/<entity>/<source_project>/<run_id>.json

   No filtering, no dedup. Re-running is incremental (existing files skipped).

2. **Analyze + dedup** (:mod:`paper_plots.analyze_al`) walks the raw store,
   applies the *complete* predicate (state finished, summary scalars present,
   history at least ``n_iterations + 1`` rows), groups by combo cell, picks
   the newest complete run per ``(combo, seed)``, and writes the merged sink::

       paper_plots/.cache_al/<entity>/<sink>/<ds_key>/<method>/<strategy>/
           notion_<notion>/loss_<loss>/cal_<cal>/seed_<seed>.json

For non-uncertainty strategies the ``notion`` segment is ``notion_n_a`` --
the strategy ignores the notion field, so all variants collapse onto one cell.

Each merged record carries ``source_project`` + ``source_run_id`` so the
provenance of every cache cell remains traceable to the wandb run that
produced it.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    import wandb

_DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache_al"

_SUMMARY_KEYS = ("nauc", "final_accuracy")
_HISTORY_KEYS = ("iteration", "labeled_size", "test_accuracy")
_HISTORY_INT_KEYS = frozenset({"iteration", "labeled_size"})

_DEFAULT_NOTION = "epistemic"
_DEFAULT_LOSS = "cross_entropy"
_DEFAULT_CAL = "none"
_NOTION_NA = "n_a"

# Classification status codes returned by :func:`classify_raw_record`.
STATUS_COMPLETE = "complete"
STATUS_WRONG_STATE = "wrong_state"
STATUS_MISSING_SUMMARY = "missing_summary"
STATUS_MISSING_N_ITER = "missing_n_iterations"
STATUS_TRUNCATED = "truncated_history"
STATUS_MISSING_AXES = "missing_combo_axes"


def cache_root() -> Path:
    """Return the AL cache root, honouring ``$PAPER_PLOTS_AL_CACHE`` if set."""
    env = os.environ.get("PAPER_PLOTS_AL_CACHE")
    return Path(env).expanduser() if env else _DEFAULT_CACHE_DIR


# ---------------------------------------------------------------------------
# Cache-cell axis helpers
# ---------------------------------------------------------------------------


def effective_notion(strategy: str, notion: str | None) -> str:
    """Notion only matters for the ``uncertainty`` strategy; collapse otherwise."""
    if strategy == "uncertainty":
        return notion or _DEFAULT_NOTION
    return _NOTION_NA


def dataset_key(name: str, openml_id: int | str | None) -> str:
    """Path-safe dataset key. ``openml`` needs the integer task id."""
    if name == "openml":
        if openml_id is None:
            msg = "openml dataset entry must include an 'openml_id'."
            raise ValueError(msg)
        return f"openml_{openml_id}"
    return name


def _config_axes(
    config: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Pull (method, strategy, notion, loss, calibration) from a run's config dict."""
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
    return method, strategy, notion, loss, calibration


def _calibration_segment(calibration: str | None) -> str:
    return f"cal_{calibration or _DEFAULT_CAL}"


def _notion_segment(notion: str) -> str:
    return f"notion_{notion or _DEFAULT_NOTION}"


def _loss_segment(supervised_loss: str | None) -> str:
    return f"loss_{supervised_loss or _DEFAULT_LOSS}"


def cell_dir(
    entity: str,
    sink: str,
    ds_key: str,
    method: str,
    strategy: str,
    notion: str | None,
    supervised_loss: str | None,
    calibration: str | None,
) -> Path:
    """Directory holding seed-level cache files for one ``(combo)`` cell."""
    return (
        cache_root()
        / entity
        / sink
        / ds_key
        / method
        / strategy
        / _notion_segment(effective_notion(strategy, notion))
        / _loss_segment(supervised_loss)
        / _calibration_segment(calibration)
    )


# ---------------------------------------------------------------------------
# JSON encoder fallback (numpy + wandb SubDict)
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:  # noqa: ANN401
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "items"):
        return dict(obj.items())
    msg = f"Cannot JSON-serialize value of type {type(obj).__name__}: {obj!r}"
    raise TypeError(msg)


def _atomic_write(path: Path, payload: Any) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(payload, f, default=_json_default)
        tmp = Path(f.name)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Raw bulk store
# ---------------------------------------------------------------------------


def raw_dir(entity: str, source_project: str) -> Path:
    return cache_root() / "_raw" / entity / source_project


def raw_record_path(entity: str, source_project: str, run_id: str) -> Path:
    return raw_dir(entity, source_project) / f"{run_id}.json"


def _config_to_jsonsafe(run_config: Any) -> dict[str, Any]:  # noqa: ANN401
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
    """Return AL history rows with the keys we care about (sorted by iteration)."""
    df = run.history(samples=1000, pandas=True, keys=list(_HISTORY_KEYS))
    if df is None or len(df) == 0:
        return []
    if not isinstance(df, pd.DataFrame):
        return []
    columns = set(df.columns)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item: dict[str, Any] = {}
        for key in _HISTORY_KEYS:
            if key not in columns:
                continue
            value = row[key]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            try:
                item[key] = int(value) if key in _HISTORY_INT_KEYS else float(value)
            except (TypeError, ValueError):
                continue
        if "iteration" in item:
            rows.append(item)
    rows.sort(key=lambda r: r.get("iteration", 0))
    return rows


def build_raw_record(run: wandb.apis.public.Run, entity: str, source_project: str) -> dict[str, Any]:
    """Build a raw record from one wandb run. Always succeeds (no filtering)."""
    return {
        "run_id": run.id,
        "name": run.name,
        "entity": entity,
        "source_project": source_project,
        "state": run.state,
        "created_at": str(run.created_at),
        "fetched_at": datetime.now(UTC).isoformat(),
        "config": _config_to_jsonsafe(run.config),
        "summary": _summary_from_run(run),
        "history": _history_from_run(run),
    }


def save_raw_record(record: dict[str, Any]) -> None:
    """Write one raw record to the bulk store."""
    path = raw_record_path(record["entity"], record["source_project"], record["run_id"])
    _atomic_write(path, record)


def iter_raw_records(entity: str, source_project: str) -> Iterable[dict[str, Any]]:
    """Yield every raw record cached for ``<entity>/<source_project>``."""
    directory = raw_dir(entity, source_project)
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        with path.open() as fh:
            yield json.load(fh)


# ---------------------------------------------------------------------------
# Classification (analyze stage)
# ---------------------------------------------------------------------------


def _check_summary(summary: dict[str, Any]) -> bool:
    """``True`` iff ``nauc`` is present and parseable.

    ``final_accuracy`` is intentionally not required: the bar/table/curves
    consumers only need ``nauc``, and many otherwise-usable runs lack the
    final-accuracy scalar.
    """
    nauc = summary.get("nauc")
    if nauc is None:
        return False
    try:
        float(nauc)
    except (TypeError, ValueError):
        return False
    return True


def _extract_combo(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Pull ``(ds_key, method, strategy, notion, supervised_loss, calibration, seed)``."""
    method, strategy, notion, loss, calibration = _config_axes(cfg)
    if not isinstance(method, str) or not isinstance(strategy, str):
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

    Acceptance rule:

    1. ``state == "finished"``.
    2. ``summary["nauc"]`` is present and parseable. ``final_accuracy`` is *not*
       required (no consumer reads it).
    3. ``config["n_iterations"]`` parseable.
    4. ``len(history) >= n_iterations + 1`` -- partial-history runs are rejected
       so curves and per-iteration consumers always see full trajectories.
       Partial *coverage at the cell level* (some seeds missing) is fine -- it's
       handled per-(cell, seed) at render time.
    5. The combo axes (dataset, method, strategy, seed, ...) are extractable.

    ``combo`` is set only when ``status == STATUS_COMPLETE``.
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


# ---------------------------------------------------------------------------
# Merged sink
# ---------------------------------------------------------------------------


def save_merged_record(*, entity: str, sink: str, combo: dict[str, Any], record: dict[str, Any]) -> Path:
    """Write the chosen newest record into the merged sink under its combo cell."""
    directory = cell_dir(
        entity,
        sink,
        combo["ds_key"],
        combo["method"],
        combo["strategy"],
        combo.get("notion"),
        combo.get("supervised_loss"),
        combo.get("calibration"),
    )
    path = directory / f"seed_{combo['seed']}.json"
    _atomic_write(path, record)
    return path


def load_merged(
    entity: str,
    sink: str,
    ds_key: str,
    method: str,
    strategy: str,
    notion: str | None = None,
    supervised_loss: str | None = None,
    calibration: str | None = None,
    seeds: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Load merged records for one ``(combo)`` cell, optionally filtered by seed."""
    directory = cell_dir(entity, sink, ds_key, method, strategy, notion, supervised_loss, calibration)
    if not directory.exists():
        return []
    seed_set: set[Any] | None = set(seeds) if seeds is not None else None
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("seed_*.json")):
        record = json.loads(path.read_text())
        seed = record.get("config", {}).get("seed")
        if seed_set is not None and seed not in seed_set:
            continue
        out.append(record)
    out.sort(key=lambda r: str(r.get("config", {}).get("seed")))
    return out


def to_merged_record(raw_record: dict[str, Any], sink: str) -> dict[str, Any]:
    """Adapt a raw record into the merged-sink record shape."""
    merged = dict(raw_record)
    merged["sink"] = sink
    merged["source_project"] = raw_record["source_project"]
    merged["source_run_id"] = raw_record["run_id"]
    return merged
