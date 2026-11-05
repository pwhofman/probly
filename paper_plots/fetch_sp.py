"""Fetch every SP run for the configured ``TARGETS`` and write to the cache.

Discovers methods and seeds automatically from wandb. Loops over every
``(project, dataset, base_model)`` triple in :mod:`paper_plots.targets`. To
refresh the cache::

    uv run python -m paper_plots.fetch_sp
"""

from __future__ import annotations

import wandb

from paper_plots import cache, method_filters
from paper_plots.targets import ENTITY, TARGETS, Target

MEASURE = "default"
DECOMPOSITION = "total"


def _fetch_one(api: wandb.Api, target: Target) -> int:
    auroc_key = f"sp/{MEASURE}/{DECOMPOSITION}/auroc"
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
        if auroc_key not in summary_keys:
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
        seen.add(key)
        n_cached += 1
        print(f"cached sp/{target.dataset}/{method}/seed_{seed}")

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
