"""Fetch every OOD run for the configured ``TARGETS`` and write to the cache.

Discovers methods and seeds automatically from wandb. Score-array keys
(``id_scores``, ``ood_scores``) that wandb stores as ``large-array`` stubs are
materialized via two fallbacks:

1. Download the run's ``wandb-summary.json`` file -- wandb often stores the
   full array there even when the public-API summary returns a placeholder.
2. If the JSON file also has a placeholder, download the matching
   ``id_scores-...`` / ``ood_scores-...`` artifact.

Loops over every ``(project, dataset, base_model)`` triple in
:mod:`paper_plots.targets`. To refresh::

    uv run python -m paper_plots.fetch_ood
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import warnings

import numpy as np
import wandb

from paper_plots import cache, method_filters, ood_groups
from paper_plots.targets import ENTITY, TARGETS, Target

MEASURE = "default"
DECOMPOSITION = "epistemic"


def _try_artifact_array(api: wandb.Api, project: str, artifact_name: str, npy_filename: str) -> np.ndarray | None:
    """Download ``<entity>/<project>/<artifact_name>:latest`` and load one ``.npy``.

    Returns ``None`` on any failure (artifact missing, network error, permission
    denied on wandb's local artifact cache, etc.) and emits a one-line warning
    so silent placeholders are diagnosable.
    """
    qualname = f"{ENTITY}/{project}/{artifact_name}:latest"
    try:
        art = api.artifact(qualname)
        with tempfile.TemporaryDirectory() as td:
            art.download(root=td)
            return np.asarray(np.load(Path(td) / npy_filename))
    except (wandb.errors.CommError, FileNotFoundError, OSError, ValueError) as exc:
        warnings.warn(
            f"artifact download failed for {qualname}: {type(exc).__name__}: {exc}",
            stacklevel=2,
        )
        return None


def _full_summary_from_run(run: wandb.apis.public.Run) -> dict[str, Any] | None:
    """Download the run's ``wandb-summary.json`` file -- often holds full arrays.

    Returns ``None`` if the file isn't available.
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            run.file("wandb-summary.json").download(replace=True, root=td)
            with (Path(td) / "wandb-summary.json").open() as fh:
                return json.load(fh)
    except (wandb.errors.CommError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None


def _resolve_from_full(value: Any) -> np.ndarray | None:  # noqa: ANN401
    """Coerce a value extracted from ``wandb-summary.json`` to an ndarray, or ``None``."""
    if value is None or cache.is_large_array_placeholder(value):
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.size > 0 else None


def _materialize_score_arrays(
    api: wandb.Api,
    run: wandb.apis.public.Run,
    target: Target,
    *,
    method: str,
    seed: object,
    summary: dict[str, Any],
) -> None:
    """Replace large-array placeholders in ``summary`` with real arrays.

    Tries ``wandb-summary.json`` first (often has the full inline value even
    when the public-API summary returns a placeholder), then falls back to the
    matching artifact.
    """
    id_key = f"ood/{MEASURE}/{DECOMPOSITION}/id_scores"
    expected_ood = set(ood_groups.NEAR_OOD.get(target.dataset, ())) | set(ood_groups.FAR_OOD.get(target.dataset, ()))
    ood_keys = {ood_ds: f"ood/{ood_ds}/{MEASURE}/{DECOMPOSITION}/ood_scores" for ood_ds in expected_ood}

    needs_resolution = cache.is_large_array_placeholder(summary.get(id_key)) or summary.get(id_key) is None
    needs_resolution = needs_resolution or any(
        cache.is_large_array_placeholder(summary.get(k)) or summary.get(k) is None for k in ood_keys.values()
    )
    full = _full_summary_from_run(run) if needs_resolution else None

    if cache.is_large_array_placeholder(summary.get(id_key)) or summary.get(id_key) is None:
        resolved = _resolve_from_full(full.get(id_key)) if full else None
        if resolved is not None:
            summary[id_key] = resolved
        else:
            arr = _try_artifact_array(
                api,
                target.project,
                f"id_scores-{method}-{target.dataset}-{MEASURE}-{DECOMPOSITION}-seed{seed}",
                "id_scores.npy",
            )
            if arr is not None:
                summary[id_key] = arr

    for ood_ds, ood_key in ood_keys.items():
        if ood_key in summary and not cache.is_large_array_placeholder(summary[ood_key]):
            continue
        resolved = _resolve_from_full(full.get(ood_key)) if full else None
        if resolved is not None:
            summary[ood_key] = resolved
            continue
        arr = _try_artifact_array(
            api,
            target.project,
            f"ood_scores-{method}-{target.dataset}-{ood_ds}-{MEASURE}-{DECOMPOSITION}-seed{seed}",
            "ood_scores.npy",
        )
        if arr is not None:
            summary[ood_key] = arr


def _fetch_one(api: wandb.Api, target: Target) -> int:
    expected_ood = set(ood_groups.NEAR_OOD.get(target.dataset, ())) | set(ood_groups.FAR_OOD.get(target.dataset, ()))
    expected_keys = {f"ood/{ds}/{MEASURE}/{DECOMPOSITION}/auroc" for ds in expected_ood}
    if not expected_keys:
        print(f"  !! no OOD groups defined for {target.dataset}; skipping")
        return 0

    runs = api.runs(
        f"{ENTITY}/{target.project}",
        filters={
            "config.dataset": target.dataset,
            "config.base_model": target.base_model,
            "state": "finished",
        },
        order="-created_at",
    )

    seen: set[tuple[str, object]] = set()
    n_cached = 0
    for run in runs:
        method_cfg = run.config.get("method")
        method = method_cfg.get("name") if isinstance(method_cfg, dict) else None
        seed = run.config.get("seed")
        if not isinstance(method, str) or seed is None:
            continue
        key = (method, seed)
        if key in seen:
            continue
        if not method_filters.passes(method, run.config):
            continue

        summary_keys = [k for k in run.summary.keys() if cache.is_relevant_key(k)]  # noqa: SIM118
        if not expected_keys.intersection(summary_keys):
            continue
        summary = {k: run.summary.get(k) for k in summary_keys}
        _materialize_score_arrays(api, run, target, method=method, seed=seed, summary=summary)

        cache.save_record(
            entity=ENTITY,
            project=target.project,
            dataset=target.dataset,
            base_model=target.base_model,
            method=method,
            seed=seed,
            run_id=run.id,
            name=run.name,
            config={
                "seed": seed,
                "dataset": target.dataset,
                "base_model": target.base_model,
                "method": {"name": method},
                "_param_filters": dict(method_filters.METHOD_PARAM_FILTERS.get(method, {})),
            },
            summary=summary,
        )
        seen.add(key)
        n_cached += 1
        print(f"cached ood/{target.dataset}/{method}/seed_{seed}")

    print(f"  -> {target.dataset}/{target.base_model}: {n_cached} runs across {len({m for m, _ in seen})} methods\n")
    return n_cached


def main() -> None:
    api = wandb.Api(timeout=60)
    total = 0
    for target in TARGETS:
        print(f"[{target.project}] {target.dataset}/{target.base_model}")
        total += _fetch_one(api, target)
    print(f"done: {total} runs cached across {len(TARGETS)} target(s).")


if __name__ == "__main__":
    main()
