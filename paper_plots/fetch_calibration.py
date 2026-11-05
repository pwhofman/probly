"""Fetch calibration-variant runs and merge them into each target's cache.

Calibration runs all use ``method.name=base`` and live under
``wandb.project=calibration``. They differ along two axes:

- ``supervised_loss.name``: ``label_smoothing`` / ``label_relaxation`` (vs. the
  default ``cross_entropy``).
- ``calibration.name``: ``vector_scaling`` / ``temperature_scaling`` (vs. the
  default ``none``).

Each variant is cached as a synthetic method name (e.g. ``base_label_smoothing``)
under the same ``(dataset, base_model)`` cache dir as the regular runs, so the
existing ``bar_sp`` / ``bar_ood`` / ``paper_tables`` pick them up with no extra
plumbing. OOD score-array placeholders are materialized via the same artifact
fallback as :mod:`paper_plots.fetch_ood`.

Run::

    uv run python -m paper_plots.fetch_calibration
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import warnings

import numpy as np
import wandb

from paper_plots import cache, method_filters, ood_groups
from paper_plots.targets import ENTITY, TARGETS, Target

CALIBRATION_PROJECT = "calibration"
SP_MEASURE = "default"
SP_DECOMPOSITION = "total"
OOD_MEASURE = "default"
OOD_DECOMPOSITION = "epistemic"


# Each variant becomes a synthetic method name. ``filters`` is merged into the
# wandb runs query; ``params_check`` is verified client-side (run is skipped if
# any value mismatches).
VARIANTS: list[dict[str, Any]] = [
    {
        "method": "base_label_smoothing",
        "filters": {
            "config.method.name": "base",
            "config.supervised_loss.name": "label_smoothing",
        },
        "params_check": {"supervised_loss.params.epsilon": 0.05},
    },
    {
        "method": "base_label_relaxation",
        "filters": {
            "config.method.name": "base",
            "config.supervised_loss.name": "label_relaxation",
        },
        "params_check": {"supervised_loss.params.alpha": 0.05},
    },
    {
        "method": "base_vector_scaling",
        "filters": {
            "config.method.name": "base",
            "config.calibration.name": "vector_scaling",
        },
        "params_check": {},
    },
    {
        "method": "base_temperature_scaling",
        "filters": {
            "config.method.name": "base",
            "config.calibration.name": "temperature_scaling",
        },
        "params_check": {},
    },
]


def _try_artifact_array(api: wandb.Api, project: str, artifact_name: str, npy_filename: str) -> np.ndarray | None:
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


def _materialize_ood_arrays(
    api: wandb.Api,
    target: Target,
    *,
    method: str,
    seed: object,
    summary: dict[str, Any],
) -> None:
    """Same artifact-fallback logic as ``fetch_ood``, but pulls from the calibration project."""
    id_key = f"ood/{OOD_MEASURE}/{OOD_DECOMPOSITION}/id_scores"
    if cache.is_large_array_placeholder(summary.get(id_key)) or summary.get(id_key) is None:
        arr = _try_artifact_array(
            api,
            CALIBRATION_PROJECT,
            f"id_scores-{method}-{target.dataset}-{OOD_MEASURE}-{OOD_DECOMPOSITION}-seed{seed}",
            "id_scores.npy",
        )
        if arr is not None:
            summary[id_key] = arr

    expected_ood = set(ood_groups.NEAR_OOD.get(target.dataset, ())) | set(ood_groups.FAR_OOD.get(target.dataset, ()))
    for ood_ds in expected_ood:
        ood_key = f"ood/{ood_ds}/{OOD_MEASURE}/{OOD_DECOMPOSITION}/ood_scores"
        if ood_key in summary and not cache.is_large_array_placeholder(summary[ood_key]):
            continue
        arr = _try_artifact_array(
            api,
            CALIBRATION_PROJECT,
            f"ood_scores-{method}-{target.dataset}-{ood_ds}-{OOD_MEASURE}-{OOD_DECOMPOSITION}-seed{seed}",
            "ood_scores.npy",
        )
        if arr is not None:
            summary[ood_key] = arr


def _fetch_variant(api: wandb.Api, target: Target, variant: dict[str, Any]) -> int:
    method = variant["method"]
    sp_key = f"sp/{SP_MEASURE}/{SP_DECOMPOSITION}/auroc"
    expected_ood = set(ood_groups.NEAR_OOD.get(target.dataset, ())) | set(ood_groups.FAR_OOD.get(target.dataset, ()))
    expected_ood_keys = {f"ood/{ds}/{OOD_MEASURE}/{OOD_DECOMPOSITION}/auroc" for ds in expected_ood}

    filters: dict[str, Any] = {
        "config.dataset": target.dataset,
        "config.base_model": target.base_model,
        "state": "finished",
    }
    filters.update(variant["filters"])

    runs = api.runs(
        f"{ENTITY}/{CALIBRATION_PROJECT}",
        filters=filters,
        order="-created_at",
    )

    seen: set[object] = set()
    n_cached = 0
    for run in runs:
        seed = run.config.get("seed")
        if seed is None or seed in seen:
            continue
        if not method_filters.params_match(run.config, variant["params_check"]):
            continue

        summary_keys = [k for k in run.summary.keys() if cache.is_relevant_key(k)]  # noqa: SIM118
        has_sp = sp_key in summary_keys
        has_ood = bool(expected_ood_keys.intersection(summary_keys))
        if not (has_sp or has_ood):
            continue
        summary = {k: run.summary.get(k) for k in summary_keys}
        if has_ood:
            _materialize_ood_arrays(api, target, method=method, seed=seed, summary=summary)

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
                "source_project": CALIBRATION_PROJECT,
            },
            summary=summary,
        )
        seen.add(seed)
        n_cached += 1
        print(f"cached calib/{target.dataset}/{method}/seed_{seed}")

    if n_cached == 0:
        warnings.warn(
            f"no runs cached for variant={method!r} on {target.dataset}/{target.base_model}",
            stacklevel=2,
        )
    return n_cached


def main() -> None:
    api = wandb.Api(timeout=60)
    total = 0
    for target in TARGETS:
        print(f"[{CALIBRATION_PROJECT}] {target.dataset}/{target.base_model}")
        for variant in VARIANTS:
            total += _fetch_variant(api, target, variant)
    print(f"done: {total} calibration runs cached across {len(TARGETS)} target(s).")


if __name__ == "__main__":
    main()
