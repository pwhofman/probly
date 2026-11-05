"""Classify the raw AL store, dedup by combo cell, write the merged sink.

Pure-local pass: no wandb traffic. Walks every raw record written by
:mod:`paper_plots.fetch_al`, applies the *complete* predicate (state finished,
``nauc`` + ``final_accuracy`` summary scalars present, history at least
``n_iterations + 1`` rows long), groups by the cache cell::

    (ds_key, method, strategy, effective_notion, loss, calibration, seed)

and keeps the **newest complete run** per cell. Each chosen record is written
under::

    paper_plots/.cache_al/<entity>/<sink>/<ds_key>/<method>/<strategy>/
        notion_<n>/loss_<l>/cal_<c>/seed_<s>.json

Re-run after every ``fetch_al`` to refresh the merged store. Idempotent and
fast (no network).

Run::

    uv run python -m paper_plots.analyze_al
"""

from __future__ import annotations

import shutil
from typing import Any

from paper_plots import al_cache
from paper_plots.targets import ENTITY

# --- edit me -----------------------------------------------------------------
SINK = "al_merged"
# Projects whose runs feed the figures. Other projects hold pilots and tests, which must
# never fill a cell just because they ran later.
AL_PROJECTS: tuple[str, ...] = (
    "al_openml_v1600_0505",
    "al_openml_v1342_0605",
    "al_openml_v1612_0605",
    "crv-al-openml",
    "crv-al-cifar10",
)
# Campaigns that redo a method after a fix. A method they cover loses all its runs from the
# other projects, including strategies they did not redo: those belong to the unfixed model
# and must not sit next to the fixed one in a figure.
SUPERSEDING_PROJECTS: frozenset[str] = frozenset({"crv-al-openml"})
ALLOWED_DATASETS: tuple[str, ...] = ("openml_6", "openml_156", "cifar10")
# -----------------------------------------------------------------------------


def _combo_cell(combo: dict[str, Any]) -> tuple[str, ...]:
    """Stable cache-cell identity (without seed)."""
    return (
        combo["ds_key"],
        combo["method"],
        combo["strategy"],
        al_cache.effective_notion(combo["strategy"], combo.get("notion")),
        combo.get("supervised_loss") or "cross_entropy",
        combo.get("calibration") or "none",
    )


def _group_raw_records() -> tuple[
    dict[tuple[str, ...], dict[Any, list[tuple[str, dict[str, Any], dict[str, Any]]]]],
    dict[str, Any],
]:
    """Walk every raw record and group complete ones by cache cell.

    Returns ``(grouped, stats)`` where ``grouped[cell][seed]`` is a list of
    ``(created_at, raw_record, combo)`` tuples (all complete, multiple wins
    possible per (cell, seed) before dedup), and ``stats`` carries
    diagnostic counters.
    """
    grouped: dict[tuple[str, ...], dict[Any, list[tuple[str, dict[str, Any], dict[str, Any]]]]] = {}
    drop_counts: dict[str, int] = {}
    total_raw = 0
    filtered_dataset = 0

    for project in AL_PROJECTS:
        n_in_project = 0
        for raw in al_cache.iter_raw_records(ENTITY, project):
            n_in_project += 1
            total_raw += 1
            status, combo = al_cache.classify_raw_record(raw)
            if combo is None:
                drop_counts[status] = drop_counts.get(status, 0) + 1
                continue
            if ALLOWED_DATASETS and combo["ds_key"] not in ALLOWED_DATASETS:
                filtered_dataset += 1
                continue
            cell = _combo_cell(combo)
            grouped.setdefault(cell, {}).setdefault(combo["seed"], []).append(
                (raw.get("created_at", ""), raw, combo),
            )
        print(f"  {project:<28s}  {n_in_project} raw records")
    print()

    return grouped, {
        "total_raw": total_raw,
        "filtered_dataset": filtered_dataset,
        "drop_counts": drop_counts,
    }


def _drop_superseded(
    grouped: dict[tuple[str, ...], dict[Any, list[tuple[str, dict[str, Any], dict[str, Any]]]]],
) -> list[tuple[str, ...]]:
    """Remove runs of redone methods from older campaigns; return the emptied cells."""

    def model(cell: tuple[str, ...]) -> tuple[str, ...]:
        ds_key, method, _strategy, _notion, loss, calibration = cell
        return ds_key, method, loss, calibration

    redone = {
        model(cell)
        for cell, by_seed in grouped.items()
        for items in by_seed.values()
        for _, raw, _ in items
        if raw["source_project"] in SUPERSEDING_PROJECTS
    }
    emptied: list[tuple[str, ...]] = []
    for cell, by_seed in list(grouped.items()):
        if model(cell) not in redone:
            continue
        for seed, items in list(by_seed.items()):
            kept = [item for item in items if item[1]["source_project"] in SUPERSEDING_PROJECTS]
            if kept:
                by_seed[seed] = kept
            else:
                del by_seed[seed]
        if not by_seed:
            del grouped[cell]
            emptied.append(cell)
    return sorted(emptied)


def _write_sink(
    grouped: dict[tuple[str, ...], dict[Any, list[tuple[str, dict[str, Any], dict[str, Any]]]]],
) -> tuple[int, dict[str, int]]:
    """Pick newest-per-(cell, seed) and write the merged record."""
    written = 0
    sources_hit: dict[str, int] = {}
    for by_seed in grouped.values():
        for items in by_seed.values():
            items.sort(key=lambda kv: kv[0], reverse=True)
            _created_at, chosen_raw, chosen_combo = items[0]
            merged = al_cache.to_merged_record(chosen_raw, sink=SINK)
            al_cache.save_merged_record(entity=ENTITY, sink=SINK, combo=chosen_combo, record=merged)
            sp = chosen_raw["source_project"]
            sources_hit[sp] = sources_hit.get(sp, 0) + 1
            written += 1
    return written, sources_hit


def main() -> None:
    print(f"AL cache root : {al_cache.cache_root()}")
    print(f"Raw store     : {al_cache.cache_root()}/_raw/{ENTITY}/<project>/")
    print(f"Sink          : {al_cache.cache_root()}/{ENTITY}/{SINK}/")
    print(f"Sources       : {list(AL_PROJECTS)}")
    print(f"Datasets      : {list(ALLOWED_DATASETS) if ALLOWED_DATASETS else 'all'}\n")

    grouped, stats = _group_raw_records()
    emptied = _drop_superseded(grouped)
    # A previous run may have written cells that no longer exist under the current rules.
    shutil.rmtree(al_cache.cache_root() / ENTITY / SINK, ignore_errors=True)
    written, sources_hit = _write_sink(grouped)

    n_complete = sum(len(seeds) for seeds in grouped.values())
    print("=== analyze summary ===")
    print(f"  raw records scanned  : {stats['total_raw']}")
    print(f"  classified COMPLETE  : {n_complete}")
    print(f"  drop breakdown       : {stats['drop_counts']}")
    print(f"  filtered by dataset  : {stats['filtered_dataset']}")
    print(f"  unique combo cells   : {len(grouped)}")
    print(f"  records written      : {written}")
    print(f"  source contributions : {sources_hit}")
    print(f"  cells only in superseded runs, dropped: {len(emptied)}")
    for cell in emptied:
        print("    " + " / ".join(cell))


if __name__ == "__main__":
    main()
