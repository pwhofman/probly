r"""Bulk-download every finished AL run from each wandb source project.

For each ``wandb.source_projects`` entry, this script queries
``state=finished`` runs once (paginated), pulls each run's config, summary,
and history, and writes the raw record to::

    probly_cache/_raw/{entity}/{source_project}/{run_id}.json

No filtering by combo is done here -- the analysis stage
(:mod:`probly_benchmark.plot.cache_analyze_al`) is responsible for
classifying records and deduping by combo.

By default the script is incremental: a raw cache file already on disk is
skipped, so re-running just picks up newly-finished runs. Set
``refresh=true`` to re-pull every run.

An optional ``openml_id_filter`` (list of int OpenML task ids) restricts
the query when the source projects also hold non-OpenML runs.

Usage::

    uv run cache_bulk_al.py
    uv run cache_bulk_al.py refresh=true
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import wandb

from probly_benchmark.paths import CACHE_PATH
from probly_benchmark.plot import cache_al


def _as_list(value: Any) -> list[Any]:  # noqa: ANN401
    if isinstance(value, ListConfig):
        materialized = OmegaConf.to_container(value, resolve=True)
        if isinstance(materialized, list):
            return list(materialized)
        return [materialized]
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _filters_for(openml_ids: list[int] | None) -> dict[str, Any]:
    """Build a wandb filter that matches every finished run we care about."""
    filters: dict[str, Any] = {"state": "finished"}
    if openml_ids:
        filters["config.dataset.name"] = "openml"
        filters["config.dataset.openml_id"] = {"$in": list(openml_ids)}
    return filters


def _bulk_one_project(
    *,
    api: wandb.Api,
    entity: str,
    source_project: str,
    filters: dict[str, Any],
    refresh: bool,
) -> tuple[int, int, int]:
    """Pull every matching run from ``source_project``; return ``(written, skipped, total)``."""
    runs = api.runs(f"{entity}/{source_project}", filters=filters, order="-created_at")
    total = len(runs)
    print(f"=== {source_project} : {total} finished runs ===")

    written = 0
    skipped = 0
    for idx, run in enumerate(runs, start=1):
        path = cache_al.raw_record_path(entity, source_project, run.id)
        if path.exists() and not refresh:
            skipped += 1
            continue
        record = cache_al.build_raw_record(run, entity, source_project)
        cache_al.save_raw_record(record)
        written += 1
        if idx % 50 == 0 or idx == total:
            print(f"  {idx}/{total}  (written={written}, skipped_existing={skipped})")
    print(f"  done: written={written}, skipped_existing={skipped}\n")
    return written, skipped, total


@hydra.main(version_base=None, config_path="../plot_configs", config_name="cache_bulk_al")
def main(cfg: DictConfig) -> None:
    """Bulk-download every finished AL run from each source project."""
    entity: str = cfg.wandb.entity
    source_projects = _as_list(cfg.wandb.get("source_projects") or [])
    if not source_projects:
        msg = "`wandb.source_projects` must list at least one source project."
        raise ValueError(msg)
    refresh = bool(cfg.get("refresh", False))
    openml_id_filter = _as_list(cfg.get("openml_id_filter") or []) or None
    if openml_id_filter is not None:
        openml_id_filter = [int(x) for x in openml_id_filter]

    print(f"Cache root: {CACHE_PATH}")
    print(f"Raw store : {CACHE_PATH}/_raw/{entity}/<source_project>/<run_id>.json")
    print(f"Sources   : {source_projects}")
    print(f"Refresh   : {refresh}")
    print(f"Filters   : openml_id in {openml_id_filter}\n")

    filters = _filters_for(openml_id_filter)
    api = wandb.Api(timeout=60)

    grand_written = 0
    grand_skipped = 0
    grand_total = 0
    for sp in source_projects:
        w, s, t = _bulk_one_project(
            api=api,
            entity=entity,
            source_project=sp,
            filters=filters,
            refresh=refresh,
        )
        grand_written += w
        grand_skipped += s
        grand_total += t

    print("=== Bulk download summary ===")
    print(f"  total runs surveyed  : {grand_total}")
    print(f"  raw records written  : {grand_written}")
    print(f"  skipped (already raw): {grand_skipped}")


if __name__ == "__main__":
    main()
