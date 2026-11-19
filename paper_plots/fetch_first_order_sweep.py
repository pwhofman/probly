"""Fetch first-order ``coverage`` / ``efficiency`` across the per-method alpha sweep.

For credal methods that have a meaningful ``method.train.alpha`` sweep
(EffCre, CreRL, CreEns), this script captures every alpha value observed in
wandb. Each ``(method, alpha, seed)`` triple lands in the dedicated sweep
cache at ``paper_plots/.cache_sweep/...`` so the regular SP/OOD cache stays
single-valued.

The ``method_filters.METHOD_PARAM_FILTERS`` pin (e.g. EffCre at alpha=0.8 for
SP/OOD/AL) is **deliberately ignored here** -- the whole point of this fetcher
is to capture every alpha. Sweep dedup is by ``(alpha, seed)`` so the most
recent run wins per cell.

Run::

    uv run python -m paper_plots.fetch_first_order_sweep
"""

from __future__ import annotations

import warnings

import wandb

from paper_plots import cache, sweep_cache
from paper_plots.fetch_first_order import ALLOWED_DATASETS
from paper_plots.targets import ENTITY, TARGETS, Target

# --- edit me -----------------------------------------------------------------
# Methods whose ``method.train.alpha`` sweep we want as a *line* in the
# coverage-efficiency plot. Methods with too few runs to span a line will fall
# back to a single scatter dot in :mod:`paper_plots.coverage_efficiency`.
SWEEP_METHODS: tuple[str, ...] = (
    "efficient_credal_prediction",
    "credal_relative_likelihood",
    "credal_ensembling",
    "base_conformal_tv",
)
# -----------------------------------------------------------------------------


def _alpha_from_run(run: wandb.apis.public.Run) -> float | None:
    """Read the per-method sweep parameter ``config.method.train.alpha``."""
    cfg = run.config or {}
    method = cfg.get("method") if isinstance(cfg, dict) else None
    train = method.get("train") if isinstance(method, dict) else None
    raw = train.get("alpha") if isinstance(train, dict) else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fetch_one(api: wandb.Api, target: Target, method: str) -> int:
    project_path = f"{ENTITY}/{target.project}"
    runs = api.runs(
        project_path,
        filters={
            "config.method.name": method,
            "config.dataset": target.dataset,
            "config.base_model": target.base_model,
            "state": "finished",
        },
        order="-created_at",
    )

    seen: set[tuple[float, object]] = set()
    n_cached = 0
    for run in runs:
        seed = run.config.get("seed")
        alpha = _alpha_from_run(run)
        if seed is None or alpha is None:
            continue
        key = (alpha, seed)
        if key in seen:
            continue

        summary_keys = [k for k in run.summary.keys() if cache.is_relevant_key(k)]  # noqa: SIM118
        if "coverage" not in summary_keys and "efficiency" not in summary_keys:
            continue
        summary = {k: run.summary.get(k) for k in summary_keys}

        sweep_cache.save_record(
            entity=ENTITY,
            project=target.project,
            dataset=target.dataset,
            base_model=target.base_model,
            method=method,
            alpha=alpha,
            seed=seed,
            run_id=run.id,
            name=run.name,
            config={
                "seed": seed,
                "dataset": target.dataset,
                "base_model": target.base_model,
                "method": {"name": method, "train": {"alpha": alpha}},
            },
            summary=summary,
        )
        seen.add(key)
        n_cached += 1
        print(f"cached fo-sweep/{target.dataset}/{method}/alpha={alpha}/seed_{seed}")

    if n_cached == 0:
        warnings.warn(
            f"no sweep runs cached for {method!r} on {target.dataset}/{target.base_model}",
            stacklevel=2,
        )
    return n_cached


def main() -> None:
    api = wandb.Api(timeout=60)
    total = 0
    for target in TARGETS:
        if target.dataset not in ALLOWED_DATASETS:
            continue
        print(f"[{target.project}] {target.dataset}/{target.base_model}")
        for method in SWEEP_METHODS:
            total += _fetch_one(api, target, method)
    print(f"done: {total} sweep runs cached.")


if __name__ == "__main__":
    main()
