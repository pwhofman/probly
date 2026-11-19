"""Audit the merged AL cache against the planned grid.

Reads only the local merged sink. For every cell required by the grid
defined in :mod:`paper_plots.al_aggregate` plus the explicit blocks below,
reports which seeds are present and which are missing. Also flags any
*unexpected* cells in the cache (configs we didn't ask for).

Run::

    uv run python -m paper_plots.al_audit

No wandb traffic. Re-run after every ``analyze_al``.
"""

from __future__ import annotations

import json
from typing import Any

from paper_plots import al_cache
from paper_plots.al_aggregate import SINK, UQ_METHODS
from paper_plots.targets import ENTITY

DATASETS: tuple[str, ...] = ("openml_6", "openml_156")
SEEDS: tuple[int, ...] = tuple(range(10))


# Each block expands into a set of expected cells. ``notion`` is interpreted
# through ``effective_notion`` so non-uncertainty strategies collapse to ``n_a``.
BLOCKS: list[dict[str, Any]] = [
    {
        "name": "Block 1 - base x {margin, least_confident, random} x {ce, ls, lr}",
        "methods": ["base"],
        "strategies": ["margin", "least_confident", "random"],
        "notions": [None],
        "losses": ["cross_entropy", "label_smoothing", "label_relaxation"],
        "calibrations": ["none"],
    },
    {
        "name": "Block 2 - 19 UQ x uncertainty x {EU, AU, TU}",
        "methods": list(UQ_METHODS),
        "strategies": ["uncertainty"],
        "notions": ["epistemic", "aleatoric", "total"],
        "losses": ["cross_entropy"],
        "calibrations": ["none"],
    },
    {
        "name": "Block 3 - 19 UQ x {random, margin}",
        "methods": list(UQ_METHODS),
        "strategies": ["random", "margin"],
        "notions": [None],
        "losses": ["cross_entropy"],
        "calibrations": ["none"],
    },
    {
        "name": "Block 4 - base x strategies x {temperature, vector} scaling",
        "methods": ["base"],
        "strategies": ["margin", "least_confident", "random"],
        "notions": [None],
        "losses": ["cross_entropy"],
        "calibrations": ["temperature_scaling", "vector_scaling"],
    },
]


CellKey = tuple[str, str, str, str, str, str]
"""(dataset, method, strategy, effective_notion, loss, calibration)."""


def _expected_cells_for_block(block: dict[str, Any]) -> set[CellKey]:
    cells: set[CellKey] = set()
    for ds in DATASETS:
        for method in block["methods"]:
            for strategy in block["strategies"]:
                for notion in block["notions"]:
                    eff = al_cache.effective_notion(strategy, notion)
                    for loss in block["losses"]:
                        for cal in block["calibrations"]:
                            cells.add((ds, method, strategy, eff, loss, cal))
    return cells


def _all_expected_cells() -> tuple[set[CellKey], dict[str, set[CellKey]]]:
    by_block: dict[str, set[CellKey]] = {}
    union: set[CellKey] = set()
    for block in BLOCKS:
        cells = _expected_cells_for_block(block)
        by_block[block["name"]] = cells
        union |= cells
    return union, by_block


def _scan_merged_sink() -> dict[CellKey, set[int]]:
    """Walk every ``seed_*.json`` and return ``{cell: {seed, ...}}``."""
    root = al_cache.cache_root() / ENTITY / SINK
    out: dict[CellKey, set[int]] = {}
    if not root.exists():
        return out
    for path in root.rglob("seed_*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cfg = record.get("config") or {}
        method, strategy, notion, loss, calibration = al_cache._config_axes(cfg)  # noqa: SLF001
        if not isinstance(method, str) or not isinstance(strategy, str):
            continue
        ds_block = cfg.get("dataset") or {}
        if isinstance(ds_block, dict):
            ds_name = ds_block.get("name")
            ds_id = ds_block.get("openml_id")
        else:
            ds_name = ds_block
            ds_id = None
        if not isinstance(ds_name, str):
            continue
        try:
            ds_key = al_cache.dataset_key(ds_name, ds_id)
        except (ValueError, KeyError):
            continue
        seed = cfg.get("seed")
        if seed is None:
            continue
        cell: CellKey = (
            ds_key,
            method,
            strategy,
            al_cache.effective_notion(strategy, notion),
            loss or "cross_entropy",
            calibration or "none",
        )
        out.setdefault(cell, set()).add(
            int(seed) if isinstance(seed, (int, str)) and str(seed).lstrip("-").isdigit() else seed
        )
    return out


def _fmt_cell(cell: CellKey) -> str:
    ds, method, strategy, notion, loss, cal = cell
    return f"{ds}/{method}/{strategy}/notion_{notion}/loss_{loss}/cal_{cal}"


def main() -> None:
    expected_union, by_block = _all_expected_cells()
    actual = _scan_merged_sink()

    print("== AL cache audit ==")
    print(f"sink path : {al_cache.cache_root() / ENTITY / SINK}")
    print(f"datasets  : {list(DATASETS)}")
    print(f"seeds     : {list(SEEDS)} ({len(SEEDS)} per cell)")
    print()

    seed_set = set(SEEDS)
    total_expected_cells = len(expected_union)
    total_expected_runs = total_expected_cells * len(seed_set)
    grand_present = sum(min(len(actual.get(cell, set()) & seed_set), len(seed_set)) for cell in expected_union)

    print("GRAND TOTAL")
    print(f"  expected cells       : {total_expected_cells}")
    print(f"  expected runs        : {total_expected_runs}")
    print(f"  cells with >= 1 seed : {sum(1 for c in expected_union if actual.get(c))}")
    print(f"  runs present         : {grand_present}")
    print(f"  coverage             : {grand_present / total_expected_runs:.1%}")
    print()

    for block in BLOCKS:
        cells = by_block[block["name"]]
        n_cells = len(cells)
        n_runs_expected = n_cells * len(seed_set)
        cells_with_runs = sum(1 for c in cells if actual.get(c))
        runs_present = sum(len(actual.get(c, set()) & seed_set) for c in cells)
        empty_cells = sorted(cells - actual.keys())
        partial_cells = sorted(
            (c for c in cells if actual.get(c) and (set(SEEDS) - actual[c])),
            key=lambda c: (-len(actual[c]), c),
        )

        print(f"--- {block['name']} ---")
        print(f"  cells expected       : {n_cells}")
        print(f"  cells with >= 1 seed : {cells_with_runs}")
        print(f"  runs expected        : {n_runs_expected}")
        print(f"  runs present         : {runs_present}")
        print(f"  coverage             : {runs_present / n_runs_expected:.1%}")
        if empty_cells:
            print(f"  cells with NO runs   : {len(empty_cells)}")
            for cell in empty_cells:
                print(f"    {_fmt_cell(cell)}")
        if partial_cells:
            print("  cells with partial seeds:")
            for cell in partial_cells:
                have = sorted(actual[cell])
                miss = sorted(seed_set - actual[cell])
                print(f"    {_fmt_cell(cell)}  have={have} miss={miss}")
        print()

    # Anything in the cache that we didn't ask for.
    extra = sorted(actual.keys() - expected_union)
    if extra:
        print("--- UNEXPECTED cells (in cache but not in any expected block) ---")
        for cell in extra:
            seeds = sorted(actual[cell])
            print(f"  {_fmt_cell(cell)}  seeds={seeds}")
    else:
        print("(no unexpected cells)")


if __name__ == "__main__":
    main()
