"""Bulk-download every finished AL run from wandb into the raw store.

Searches every project of :data:`ENTITY` plus :data:`EXTRA_SOURCES`, because a
redone run can land in a project nobody listed. A run counts as an AL run when
its config has an ``al_strategy``. Each one goes to
``paper_plots/.cache_al/_raw/<entity>/<project>/<run_id>.json`` with its config,
summary scalars (``nauc``, ``final_accuracy``), and per-iteration history.

This stage does not judge the runs: pilot and test projects are captured too,
and :mod:`paper_plots.analyze_al` decides which sources feed the figures.

Re-running is incremental, so a raw record already on disk is skipped. Pass
``--refresh`` to pull every run again::

    uv run python -m paper_plots.fetch_al --refresh
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from typing import Any

import wandb

from paper_plots import al_cache
from paper_plots.targets import ENTITY

# --- edit me -----------------------------------------------------------------
# Projects outside ENTITY that hold AL runs (also read by scripts/inspect_al_runs.py).
EXTRA_SOURCES: tuple[tuple[str, str], ...] = (("speyewear", "jakubpaplham-al"),)
MAX_WORKERS = 12  # parallel run.history() calls; bump up if wandb isn't throttling
# -----------------------------------------------------------------------------


def is_al_run(config: dict[str, Any]) -> bool:
    """Return whether a run's config describes an active learning run."""
    return bool(config.get("al_strategy"))


def _dataset_of(config: dict[str, Any]) -> str:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return str(dataset)
    if dataset.get("name") == "openml":
        return f"openml_{dataset.get('openml_id')}"
    return str(dataset.get("name"))


def _sources(api: Any) -> list[tuple[str, str]]:  # noqa: ANN401
    return [(ENTITY, project.name) for project in api.projects(ENTITY)] + list(EXTRA_SOURCES)


def _fetch_and_save(run: Any, entity: str, project: str) -> None:  # noqa: ANN401
    al_cache.save_raw_record(al_cache.build_raw_record(run, entity, project))


def _bulk_one(api: Any, entity: str, project: str, *, refresh: bool) -> tuple[int, int, int, int]:  # noqa: ANN401
    """Pull the AL runs of one project; return ``(found, written, skipped, failed)``."""
    runs = api.runs(f"{entity}/{project}", filters={"state": "finished"}, order="-created_at", per_page=500)
    al_runs = [run for run in runs if is_al_run(run.config or {})]
    if not al_runs:
        return 0, 0, 0, 0

    datasets = Counter(_dataset_of(run.config) for run in al_runs)
    newest = max(str(run.created_at) for run in al_runs)[:10]
    print(f"=== {entity}/{project}: {len(al_runs)} AL runs, newest {newest}, {dict(datasets)}", flush=True)

    todo = [run for run in al_runs if refresh or not al_cache.raw_record_path(entity, project, run.id).exists()]
    written = failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_and_save, run, entity, project): run.id for run in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
                written += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  !! {futures[future]}: {type(exc).__name__}: {exc}", flush=True)
            if done % max(50, len(todo) // 10) == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} (written={written}, failed={failed})", flush=True)
    return len(al_runs), written, len(al_runs) - len(todo), failed


def fetch(api: Any, *, refresh: bool = False) -> int:  # noqa: ANN401
    """Pull every AL run reachable through ``api``; return the number of failed runs."""
    print(f"AL cache root: {al_cache.cache_root()}", flush=True)
    totals = [0, 0, 0, 0]
    unreadable: list[str] = []
    sources = _sources(api)
    for entity, project in sources:
        try:
            counts = _bulk_one(api, entity, project, refresh=refresh)
        except (wandb.errors.CommError, ValueError) as exc:
            unreadable.append(f"{entity}/{project}: {type(exc).__name__}")
            continue
        totals = [total + count for total, count in zip(totals, counts, strict=True)]
    found, written, skipped, failed = totals
    print(f"\ndone: searched {len(sources)} projects, found {found} AL runs.", flush=True)
    print(f"written={written}, already cached={skipped}, failed={failed}", flush=True)
    if unreadable:
        print("could not read:", *unreadable, sep="\n  ", flush=True)
    return failed


def main(*, refresh: bool = False) -> int:
    """Run the bulk download against the wandb API; return the number of failed runs."""
    return fetch(wandb.Api(timeout=60), refresh=refresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download every finished AL run from wandb.")
    parser.add_argument("--refresh", action="store_true", help="Pull runs again even if already cached.")
    sys.exit(1 if main(refresh=parser.parse_args().refresh) else 0)
