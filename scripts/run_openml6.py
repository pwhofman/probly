#!/usr/bin/env python3
r"""Run the openml_6 sweep, skipping anything already finished — fully offline.

Covers all openml_6 settings (no conformal-prediction blocks):

- Baselines:        ``method=base`` x {margin, entropy, least_confident} x 10 seeds
- Calibration:      ``method=base`` + {temperature_scaling, vector_scaling}
- Supervised loss:  ``method=base`` + {label_smoothing, label_relaxation}
- UQ methods x {margin, random}
- UQ uncertainty:EU per method
- UQ uncertainty:TU per method (excluding ddu)

Determines what's already done by reading a local ``.pkl`` / ``.csv`` (defaults to
``scripts/al_analysis_out/wandb_cache_runs.pkl`` produced by
``scripts/inspect_al_runs.py``). No wandb roundtrip happens here — refresh the
cache first if you want fresh state.

A combo is considered "done" if a finished run with a matching tuple of
``(method, strategy, notion, seed, calibration, supervised_loss)`` exists in the
seed file(s) for ``dataset == openml_6``.

Usage::

    # Pre-req: populate the cache (one-time or whenever you want fresh state).
    uv run python scripts/inspect_al_runs.py --refresh

    # Dry-run: print the missing combos and a summary; takes ~1s.
    uv run python scripts/run_openml6.py

    # Actually run the missing combos sequentially (continues on failure).
    uv run python scripts/run_openml6.py --execute

    # Smoke test: run only the first 3 missing combos and stop.
    uv run python scripts/run_openml6.py --execute --limit 3

    # Use a different seed file (e.g. a hand-curated CSV).
    uv run python scripts/run_openml6.py --seed-file my_done.csv

    # Combine multiple seed files (the union counts as 'done').
    uv run python scripts/run_openml6.py \\
        --seed-file scripts/al_analysis_out/wandb_cache_runs.pkl \\
        --seed-file extra_runs.csv

Seed-file schema (CSV or pickled DataFrame):

    Required: method, strategy, seed, state, dataset
    Optional: notion, calibration, supervised_loss

Only rows with ``state == "finished"`` and ``dataset == "openml_6"`` are counted.

New runs are launched with ``wandb.project=max-test`` and ``+wandb.entity=probly``;
edit the ``WANDB_PROJECT`` / ``WANDB_ENTITY`` constants if you want to redirect.
Default is dry-run; pass ``--execute`` to actually launch missing combos.
"""
# ruff: noqa: T201, ANN401, D103

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# ---- Sweep spec --------------------------------------------------------------------

DEFAULT_SEED_FILE = "scripts/al_analysis_out/wandb_cache_runs.pkl"
DATASET = "openml_6"
SEEDS: tuple[int, ...] = tuple(range(10))
BASE_STRATEGIES: tuple[str, ...] = ("margin", "entropy", "least_confident")
# Ensemble-based methods are slow to train (multiple base models per run); they
# are placed at the tail of UQ_METHODS so the per-block iteration runs them
# last, and the global execution order also pulls them to the very end via
# ENSEMBLE_METHODS below.
ENSEMBLE_METHODS: frozenset[str] = frozenset({"ensemble", "credal_ensembling"})

UQ_METHODS: tuple[str, ...] = (
    "dropout",
    "dropconnect",
    "bayesian",
    "dare",
    "evidential_classification",
    "posterior_network",
    "credal_relative_likelihood",
    "ddu",
    # ensemble-based — kept last
    "ensemble",
    "credal_ensembling",
)
UQ_TU_METHODS: tuple[str, ...] = tuple(m for m in UQ_METHODS if m != "ddu")
CALIBRATIONS: tuple[str, ...] = ("temperature_scaling", "vector_scaling")
SUPERVISED_LOSSES: tuple[str, ...] = ("label_smoothing", "label_relaxation")

# Where to send the new runs we launch.
WANDB_PROJECT = "max-test"
WANDB_ENTITY = "probly"
DEFAULT_DEVICE = "cpu"

Combo = dict[str, Any]


# ---- Key normalization & matching --------------------------------------------------

# Per-field default values that mean "no override". Treated as None when keying so
# e.g. ``supervised_loss == "cross_entropy"`` (explicit default in wandb) matches a
# combo where supervised_loss isn't set.
_FIELD_DEFAULTS: dict[str, tuple[str, ...]] = {
    "notion": (),
    "calibration": ("none",),
    "supervised_loss": ("cross_entropy",),
    "conformal": ("none",),
}


def _norm_field(value: Any, field: str) -> str | None:
    if value is None:
        return None
    s = str(value)
    if s in _FIELD_DEFAULTS.get(field, ()):
        return None
    return s


def _key(combo: Combo) -> tuple[Any, ...]:
    return (
        str(combo["method"]),
        str(combo["strategy"]),
        _norm_field(combo.get("notion"), "notion"),
        int(combo["seed"]),
        _norm_field(combo.get("calibration"), "calibration"),
        _norm_field(combo.get("supervised_loss"), "supervised_loss"),
    )


# ---- File-based seeding (read finished runs from CSV / pickle) ---------------------


def _row_value(row: pd.Series, col: str) -> Any:
    if col not in row.index:
        return None
    v = row[col]
    return None if pd.isna(v) else v


def load_seed_file(path: Path, dataset_full: str = DATASET) -> set[tuple[Any, ...]]:
    """Load finished combos from a .pkl or .csv with the inspect_al_runs cache schema.

    Required columns: ``method``, ``strategy``, ``seed``, ``state``, ``dataset``.
    Optional: ``notion``, ``calibration``, ``supervised_loss``.
    """
    if path.suffix == ".pkl":
        df = cast("pd.DataFrame", pd.read_pickle(path))  # noqa: S301
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        msg = f"Unsupported seed-file extension: {path.suffix} ({path})"
        raise ValueError(msg)

    required = {"method", "strategy", "seed", "state", "dataset"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Seed file {path} is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    df = df[(df["state"] == "finished") & (df["dataset"] == dataset_full)]
    keys: set[tuple[Any, ...]] = set()
    for _, row in df.iterrows():
        strategy = _row_value(row, "strategy")
        notion = _row_value(row, "notion") if strategy == "uncertainty" else None
        method = _row_value(row, "method")
        seed = _row_value(row, "seed")
        if method is None or strategy is None or seed is None:
            continue
        keys.add(
            _key(
                {
                    "method": method,
                    "strategy": strategy,
                    "notion": notion,
                    "seed": int(seed),
                    "calibration": _row_value(row, "calibration"),
                    "supervised_loss": _row_value(row, "supervised_loss"),
                }
            )
        )
    print(f"  [{path}] loaded {len(keys)} finished combos for {dataset_full}")
    return keys


# ---- Block definitions -------------------------------------------------------------


def block_combos() -> Iterator[tuple[str, list[Combo]]]:
    yield (
        "Baselines (base, 3 strategies, 10 seeds)",
        [{"method": "base", "strategy": s, "seed": seed} for s in BASE_STRATEGIES for seed in SEEDS],
    )
    yield (
        f"Calibration (base + {{{', '.join(CALIBRATIONS)}}})",
        [
            {"method": "base", "strategy": s, "seed": seed, "calibration": cal}
            for cal in CALIBRATIONS
            for s in BASE_STRATEGIES
            for seed in SEEDS
        ],
    )
    yield (
        f"Supervised Loss (base + {{{', '.join(SUPERVISED_LOSSES)}}})",
        [
            {"method": "base", "strategy": s, "seed": seed, "supervised_loss": sup}
            for sup in SUPERVISED_LOSSES
            for s in BASE_STRATEGIES
            for seed in SEEDS
        ],
    )
    yield (
        "UQ (margin + random)",
        [
            {"method": m, "strategy": s, "seed": seed}
            for m in UQ_METHODS
            for s in ("margin", "random")
            for seed in SEEDS
        ],
    )
    yield (
        "UQ uncertainty:EU",
        [{"method": m, "strategy": "uncertainty", "notion": "EU", "seed": seed} for m in UQ_METHODS for seed in SEEDS],
    )
    yield (
        "UQ uncertainty:TU",
        [
            {"method": m, "strategy": "uncertainty", "notion": "TU", "seed": seed}
            for m in UQ_TU_METHODS
            for seed in SEEDS
        ],
    )


# ---- Command construction ----------------------------------------------------------


def make_command(combo: Combo, *, device: str = DEFAULT_DEVICE) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "probly_benchmark.active_learning",
        f"method={combo['method']}",
        f"al_strategy={combo['strategy']}",
        f"seed={combo['seed']}",
        f"al_dataset={DATASET}",
        "wandb.enabled=true",
        f"wandb.project={WANDB_PROJECT}",
        f"+wandb.entity={WANDB_ENTITY}",
        "save_results=false",
        f"device={device}",
    ]
    if combo.get("notion"):
        cmd.append(f"al_strategy.notion={combo['notion']}")
    if combo.get("calibration"):
        cmd.append(f"calibration={combo['calibration']}")
    if combo.get("supervised_loss"):
        cmd.append(f"supervised_loss={combo['supervised_loss']}")
    return cmd


# ---- CLI entry point ---------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:  # noqa: PLR0912
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed-file",
        action="append",
        metavar="PATH",
        help=(
            f"file(s) holding finished AL runs (.pkl or .csv with the inspect_al_runs schema). "
            f"Repeatable; the union of all files is treated as 'done'. "
            f"Default: {DEFAULT_SEED_FILE}"
        ),
    )
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--execute", action="store_true", help="run missing combos (default: dry-run)")
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop on first failed combo (default: continue and report a tally)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="when --execute, run at most N missing combos and stop.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    raw_seed_files = args.seed_file or [DEFAULT_SEED_FILE]
    seed_files = [Path(s).expanduser() for s in raw_seed_files]
    for sf in seed_files:
        if not sf.exists():
            p.error(
                f"seed file not found: {sf}\n"
                f"Run `uv run python scripts/inspect_al_runs.py` first to generate the cache, "
                f"or pass a custom --seed-file."
            )

    print(f"Reading {len(seed_files)} seed file(s) (no wandb roundtrip):")
    finished: set[tuple[Any, ...]] = set()
    for sf in seed_files:
        finished |= load_seed_file(sf)
    print(f"Total finished {DATASET} combos: {len(finished)}\n")

    missing: list[tuple[str, Combo]] = []
    grand_total = 0
    for block_name, combos in block_combos():
        block_missing = [c for c in combos if _key(c) not in finished]
        print(f"=== {block_name}: {len(block_missing)}/{len(combos)} missing ===")
        for c in block_missing:
            cmd = make_command(c, device=args.device)
            print("  $", shlex.join(cmd))
        missing.extend((block_name, c) for c in block_missing)
        grand_total += len(combos)
        print()

    print(f"Summary: {len(missing)} of {grand_total} combos still need to run.")

    if not args.execute:
        print("Dry run; pass --execute to launch them.")
        return 0

    # Stable-sort: non-ensemble combos first, ensemble-based last (across blocks).
    missing.sort(key=lambda item: item[1]["method"] in ENSEMBLE_METHODS)
    n_ensemble = sum(1 for _, c in missing if c["method"] in ENSEMBLE_METHODS)
    if n_ensemble:
        print(f"Reordered: {n_ensemble} ensemble-based combos pulled to the end.")

    if args.limit is not None:
        print(f"--limit={args.limit}: will execute at most {args.limit} of {len(missing)} combos.")
        missing = missing[: args.limit]

    failures: list[tuple[Combo, int]] = []
    for i, (block_name, combo) in enumerate(missing, start=1):
        cmd = make_command(combo, device=args.device)
        print(f"\n[{i}/{len(missing)}] {block_name}")
        print("  $", shlex.join(cmd), flush=True)
        rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
        if rc != 0:
            failures.append((combo, rc))
            print(f"  (rc={rc})")
            if args.fail_fast:
                print("--fail-fast: stopping early.")
                return rc

    succeeded = len(missing) - len(failures)
    print(f"\nDone. {succeeded} succeeded, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for combo, rc in failures:
            print(f"  rc={rc} {combo}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
