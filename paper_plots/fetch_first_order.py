"""Fetch first-order coverage / efficiency results into the existing cache.

Background: ``probly_benchmark.first_order_data`` evaluates a credal-method
artifact on the first-order ``cifar10h`` test split and writes
``summary["coverage"]`` and ``summary["efficiency"]`` onto the same wandb run
that already carries ``sp/`` / ``ood/`` keys (project ``cifar10-benchmark``).

This script picks up those scalars and merges them into the existing
``(method, seed)`` cache cell. Re-running it is idempotent: the cache record
gets re-written with the latest filtered summary (which now includes the
``coverage`` / ``efficiency`` scalars next to whatever ``sp/``/``ood/`` keys
the same run logged).

Run::

    uv run python -m paper_plots.fetch_first_order
"""

from __future__ import annotations

from typing import Any
import warnings

import wandb

from paper_plots import cache, method_filters
from paper_plots.targets import ENTITY, TARGETS, Target

# --- edit me -----------------------------------------------------------------
# Coverage/efficiency was only computed for credal methods on cifar10.
CREDAL_METHODS: tuple[str, ...] = (
    "credal_bnn",
    "credal_ensembling",
    "credal_net",
    "credal_relative_likelihood",
    "credal_wrapper",
    "efficient_credal_prediction",
)
ALLOWED_DATASETS: tuple[str, ...] = ("cifar10",)
# Optional pin on the credal-set conformal target rate. ``None`` means "accept
# whatever alpha was logged" -- dedup by (method, seed) keeps the most-recent
# run, so alphas may differ across methods. Set to a float (e.g. 0.1) to enforce
# a single canonical target rate across methods. The full alpha sweep lives in
# :mod:`paper_plots.fetch_first_order_sweep`.
CANONICAL_CONFORMAL_ALPHA: float | None = None
# -----------------------------------------------------------------------------


def _fetch_one(api: wandb.Api, target: Target, method: str) -> int:
    project_path = f"{ENTITY}/{target.project}"
    filters: dict[str, Any] = {
        "config.method.name": method,
        "config.dataset": target.dataset,
        "config.base_model": target.base_model,
        "state": "finished",
    }
    if CANONICAL_CONFORMAL_ALPHA is not None:
        filters["config.conformal.alpha"] = CANONICAL_CONFORMAL_ALPHA
    runs = api.runs(project_path, filters=filters, order="-created_at")

    seen: set[object] = set()
    n_cached = 0
    for run in runs:
        seed = run.config.get("seed")
        if seed is None or seed in seen:
            continue
        if not method_filters.passes(method, run.config):
            continue

        summary_keys = [k for k in run.summary.keys() if cache.is_relevant_key(k)]  # noqa: SIM118
        if "coverage" not in summary_keys and "efficiency" not in summary_keys:
            continue
        summary = {k: run.summary.get(k) for k in summary_keys}

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
        seen.add(seed)
        n_cached += 1
        print(f"cached fo/{target.dataset}/{method}/seed_{seed}")

    if n_cached == 0:
        warnings.warn(
            f"no first-order runs cached for {method!r} on {target.dataset}/{target.base_model}",
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
        for method in CREDAL_METHODS:
            total += _fetch_one(api, target, method)
    print(f"done: {total} first-order runs cached.")


if __name__ == "__main__":
    main()
