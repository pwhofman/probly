r"""Analyze the raw AL cache and write the merged, deduped sink.

This is a pure-local pass: no wandb traffic. It walks every raw record
written by :mod:`probly_benchmark.plot.cache_bulk_al`, applies the
*complete* predicate (state finished, summary scalars present, history at
least ``n_iterations + 1`` rows long), groups by the cache combo
``(ds_key, method, strategy, notion, loss, calibration, seed)``, and keeps
the newest complete run per combo.

Each kept record is written under::

    probly_cache/{entity}/{cache.sink}/{ds_key}/{method}/{strategy}/
        notion_{notion}/loss_{loss}/cal_{cal}/seed_{seed}.json

The script is idempotent and fast (re-run after every ``cache_bulk_al.py``
without re-downloading from wandb).

Usage::

    uv run cache_analyze_al.py
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

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


def _combo_key(combo: dict[str, Any]) -> tuple[str, ...]:
    """Stable cache-cell identity (without seed)."""
    return (
        combo["ds_key"],
        combo["method"],
        combo["strategy"],
        cache_al.effective_notion(combo["strategy"], combo.get("notion")),
        combo.get("supervised_loss") or "cross_entropy",
        combo.get("calibration") or "none",
    )


def _group_raw_records(
    entity: str,
    source_projects: list[str],
    allowed: set[str] | None,
) -> tuple[dict[tuple[str, ...], dict[Any, list[tuple[str, dict[str, Any]]]]], dict[str, Any]]:
    """Walk every raw record and group complete ones by cache cell.

    Returns ``(grouped, stats)``. ``grouped[cell][seed]`` is a list of
    ``(created_at, raw_record)`` pairs, all complete. ``stats`` carries the
    ``total_raw``, ``filtered_dataset``, and per-status ``drop_counts``.
    """
    grouped: dict[tuple[str, ...], dict[Any, list[tuple[str, dict[str, Any]]]]] = {}
    drop_counts: dict[str, int] = {}
    total_raw = 0
    filtered_dataset = 0

    for sp in source_projects:
        n_in_project = 0
        for raw in cache_al.iter_raw_records(entity, sp):
            n_in_project += 1
            total_raw += 1
            status, combo = cache_al.classify_raw_record(raw)
            if combo is None:
                drop_counts[status] = drop_counts.get(status, 0) + 1
                continue
            if allowed is not None and combo["ds_key"] not in allowed:
                filtered_dataset += 1
                continue
            cell = _combo_key(combo)
            grouped.setdefault(cell, {}).setdefault(combo["seed"], []).append((raw.get("created_at", ""), raw))
        print(f"  {sp:<32s} : {n_in_project} raw records")
    print()

    return grouped, {"total_raw": total_raw, "filtered_dataset": filtered_dataset, "drop_counts": drop_counts}


def _write_sink(
    grouped: dict[tuple[str, ...], dict[Any, list[tuple[str, dict[str, Any]]]]],
    sink: str,
) -> tuple[int, dict[str, int]]:
    """Pick newest-per-seed in every cell and write to the merged sink."""
    written = 0
    sources_hit: dict[str, int] = {}
    for by_seed in grouped.values():
        for items in by_seed.values():
            items.sort(key=lambda kv: kv[0], reverse=True)
            chosen_raw = items[0][1]
            sink_record = cache_al.to_sink_record(chosen_raw, sink_project=sink)
            cache_al.save_run(sink_record)
            sources_hit[chosen_raw["source_project"]] = sources_hit.get(chosen_raw["source_project"], 0) + 1
            written += 1
    return written, sources_hit


@hydra.main(version_base=None, config_path="../plot_configs", config_name="cache_analyze_al")
def main(cfg: DictConfig) -> None:
    """Read raw store, classify + dedup, write merged sink."""
    entity: str = cfg.wandb.entity
    source_projects = _as_list(cfg.wandb.get("source_projects") or [])
    if not source_projects:
        msg = "`wandb.source_projects` must list at least one source project."
        raise ValueError(msg)
    sink: str = cfg.cache.sink
    allowed = set(_as_list(cfg.get("allowed_datasets") or [])) or None

    print(f"Cache root  : {CACHE_PATH}")
    print(f"Raw store   : {CACHE_PATH}/_raw/{entity}/<source_project>/")
    print(f"Sink        : {CACHE_PATH}/{entity}/{sink}/")
    print(f"Sources     : {source_projects}")
    print(f"Datasets    : {sorted(allowed) if allowed else 'all'}\n")

    grouped, group_stats = _group_raw_records(entity, source_projects, allowed)
    written, sources_hit = _write_sink(grouped, sink)

    n_complete = sum(len(seeds) for seeds in grouped.values())
    print("=== Analyze summary ===")
    print(f"  raw records scanned  : {group_stats['total_raw']}")
    print(f"  classified COMPLETE  : {n_complete}")
    print(f"  drop breakdown       : {group_stats['drop_counts']}")
    print(f"  filtered by dataset  : {group_stats['filtered_dataset']}")
    print(f"  unique combo cells   : {len(grouped)}")
    print(f"  records written      : {written}")
    print(f"  source contributions : {sources_hit}")
    print(f"Sink path: {CACHE_PATH}/{entity}/{sink}/")


if __name__ == "__main__":
    main()
