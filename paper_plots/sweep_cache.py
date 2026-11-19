"""Local cache for first-order alpha-sweep data.

Separate from the SP/OOD cache so the per-(method, alpha, seed) cells don't
pollute :mod:`paper_plots.bar_sp`, :mod:`paper_plots.bar_ood`, or
:mod:`paper_plots.paper_tables`.

Layout::

    paper_plots/.cache_sweep/<entity>/<project>/<dataset>/<base_model>/
        <method>/alpha_<value>/seed_<seed>.json
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

_DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache_sweep"


def _json_default(obj: Any) -> Any:  # noqa: ANN401
    """JSON encoder fallback for numpy values and wandb SummarySubDicts."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "items"):
        return dict(obj.items())
    msg = f"Cannot JSON-serialize value of type {type(obj).__name__}: {obj!r}"
    raise TypeError(msg)


def cache_root() -> Path:
    env = os.environ.get("PAPER_PLOTS_SWEEP_CACHE")
    return Path(env).expanduser() if env else _DEFAULT_CACHE_DIR


def _alpha_segment(alpha: float | str) -> str:
    """Filesystem-safe alpha directory name."""
    return f"alpha_{float(alpha):g}"


def _alpha_dir(entity: str, project: str, dataset: str, base_model: str, method: str, alpha: float | str) -> Path:
    return cache_root() / entity / project / dataset / base_model / method / _alpha_segment(alpha)


def save_record(
    *,
    entity: str,
    project: str,
    dataset: str,
    base_model: str,
    method: str,
    alpha: float | str,
    seed: int | str,
    run_id: str,
    name: str,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    """Write a single sweep record atomically. Returns the file path."""
    directory = _alpha_dir(entity, project, dataset, base_model, method, alpha)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed_{seed}.json"
    payload = {
        "run_id": run_id,
        "name": name,
        "config": config,
        "summary": summary,
        "cached_at": datetime.now(UTC).isoformat(),
    }
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(payload, f, default=_json_default)
        tmp = Path(f.name)
    tmp.replace(path)
    return path


def list_alphas(entity: str, project: str, dataset: str, base_model: str, method: str) -> list[float]:
    """Return cached alpha values for ``method`` in ascending order."""
    method_dir = cache_root() / entity / project / dataset / base_model / method
    if not method_dir.exists():
        return []
    out: list[float] = []
    for sub in method_dir.iterdir():
        if not sub.is_dir() or not sub.name.startswith("alpha_"):
            continue
        try:
            out.append(float(sub.name.removeprefix("alpha_")))
        except ValueError:
            continue
    return sorted(out)


def load_records(
    entity: str,
    project: str,
    dataset: str,
    base_model: str,
    method: str,
    alpha: float | str,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Load sweep records for one ``(method, alpha)`` cell, optionally filtered by seed."""
    directory = _alpha_dir(entity, project, dataset, base_model, method, alpha)
    if not directory.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("seed_*.json")):
        record = json.loads(path.read_text())
        if seeds is not None:
            seed = record.get("config", {}).get("seed")
            if seed not in seeds:
                continue
        out.append(record)
    return out


__all__ = [
    "cache_root",
    "list_alphas",
    "load_records",
    "save_record",
]
