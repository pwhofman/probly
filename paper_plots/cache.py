"""Local JSON cache for wandb run summaries.

Layout:
    <cache_root>/<entity>/<project>/<dataset>/<base_model>/<method>/seed_<seed>.json

Each file holds the run id, name, the relevant config subset, and the subset of
summary keys starting with ``sp/`` or ``ood/``. Fetch scripts write; bar scripts
read. There is no concept of "stale" — the user re-runs a fetch script when they
want fresh data.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

_RELEVANT_PREFIXES = ("sp/", "ood/")
# Top-level scalar keys we also keep (first-order evaluation metrics).
_RELEVANT_TOPLEVEL = frozenset({"coverage", "efficiency"})
_DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"


def _json_default(obj: Any) -> Any:  # noqa: ANN401
    """JSON encoder fallback for numpy values and wandb SummarySubDicts."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    # wandb's SummarySubDict is dict-like but not a dict subclass.
    if hasattr(obj, "items"):
        return dict(obj.items())
    msg = f"Cannot JSON-serialize value of type {type(obj).__name__}: {obj!r}"
    raise TypeError(msg)


def cache_root() -> Path:
    """Return the cache root, honouring ``$PAPER_PLOTS_CACHE`` if set."""
    env = os.environ.get("PAPER_PLOTS_CACHE")
    return Path(env).expanduser() if env else _DEFAULT_CACHE_DIR


def _method_dir(entity: str, project: str, dataset: str, base_model: str, method: str) -> Path:
    return cache_root() / entity / project / dataset / base_model / method


def _record_path(entity: str, project: str, dataset: str, base_model: str, method: str, seed: int | str) -> Path:
    return _method_dir(entity, project, dataset, base_model, method) / f"seed_{seed}.json"


def is_relevant_key(key: str) -> bool:
    """Return ``True`` for summary keys we persist (``sp/`` / ``ood/`` / coverage / efficiency)."""
    return any(key.startswith(p) for p in _RELEVANT_PREFIXES) or key in _RELEVANT_TOPLEVEL


def filter_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep only the summary keys the cache cares about."""
    return {k: v for k, v in summary.items() if is_relevant_key(k)}


def is_large_array_placeholder(value: Any) -> bool:  # noqa: ANN401
    """Return ``True`` for the wandb ``{"_type": "large-array", ...}`` stub.

    SummarySubDict isn't a dict subclass, so we duck-type on ``.get``.
    """
    getter = getattr(value, "get", None)
    if getter is None:
        return False
    try:
        return getter("_type") == "large-array"
    except TypeError:
        return False


def save_record(
    *,
    entity: str,
    project: str,
    dataset: str,
    base_model: str,
    method: str,
    seed: int | str,
    run_id: str,
    name: str,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    """Write a single run's record atomically. Returns the file path."""
    path = _record_path(entity, project, dataset, base_model, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "name": name,
        "config": config,
        "summary": summary,
        "cached_at": datetime.now(UTC).isoformat(),
    }
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(payload, f, default=_json_default)
        tmp = Path(f.name)
    tmp.replace(path)
    return path


def list_methods(entity: str, project: str, dataset: str, base_model: str) -> list[str]:
    """Return the method names that have at least one cached record."""
    base = cache_root() / entity / project / dataset / base_model
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and any(p.glob("seed_*.json")))


def load_records(
    entity: str,
    project: str,
    dataset: str,
    base_model: str,
    method: str,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Load all cached records for a method, optionally filtered to ``seeds``."""
    method_dir = _method_dir(entity, project, dataset, base_model, method)
    if not method_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(method_dir.glob("seed_*.json")):
        record = json.loads(path.read_text())
        if seeds is not None:
            seed = record.get("config", {}).get("seed")
            if seed not in seeds:
                continue
        records.append(record)
    return records
