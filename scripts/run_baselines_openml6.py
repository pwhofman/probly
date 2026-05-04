#!/usr/bin/env python3
"""Run the openml_6 baseline + UQ sweep, skipping anything already finished in wandb.

Mirrors the blocks of ``run_baselines_openml6.sh`` (baselines, calibration, supervised
loss, UQ margin/random, UQ uncertainty EU, UQ uncertainty TU) but expands each Hydra
``-m`` multirun into explicit per-combination commands and runs only those whose
``(method, strategy, notion, seed, calibration, supervised_loss)`` tuple is not
already finished in the configured wandb source(s).

Default is a dry run: it lists missing combos. Pass ``--execute`` to actually launch.

Sources of "already finished":

- ``--source ENTITY/PROJECT`` (live wandb query, default ``probly/max-test``).
- ``--seed-file PATH`` (offline) reads finished combos from a ``.pkl`` (e.g. the
  ``wandb_cache_runs.pkl`` produced by ``scripts/inspect_al_runs.py``) or a ``.csv``
  with the same schema. Useful to combine multiple wandb projects globally without
  another live query, or to mark things as done from a hand-rolled CSV.
- Both flags are repeatable; the union of their finished combos is treated as 'done'.
- Pass ``--no-wandb`` to rely solely on ``--seed-file`` inputs.
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
import wandb

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# ---- Sweep spec (mirrors run_baselines_openml6.sh) ---------------------------------

DEFAULT_FINISHED_SOURCES: tuple[str, ...] = ("probly/max-test",)
DATASET = "openml_6"
SEEDS: tuple[int, ...] = tuple(range(10))
BASE_STRATEGIES: tuple[str, ...] = ("margin", "entropy", "least_confident")
UQ_METHODS: tuple[str, ...] = (
    "dropout",
    "dropconnect",
    "bayesian",
    "dare",
    "ensemble",
    "evidential_classification",
    "posterior_network",
    "credal_ensembling",
    "credal_relative_likelihood",
    "ddu",
)
UQ_TU_METHODS: tuple[str, ...] = tuple(m for m in UQ_METHODS if m != "ddu")

WANDB_PROJECT = "max-test"
WANDB_ENTITY = "probly"
DEFAULT_DEVICE = "cpu"

Combo = dict[str, Any]


# ---- Key normalization & matching --------------------------------------------------


def _norm(value: Any) -> str | None:
    """Normalize None / 'none' / missing into None; everything else stringified."""
    if value is None:
        return None
    s = str(value)
    return None if s == "none" else s


def _key(combo: Combo) -> tuple[Any, ...]:
    return (
        str(combo["method"]),
        str(combo["strategy"]),
        _norm(combo.get("notion")),
        int(combo["seed"]),
        _norm(combo.get("calibration")),
        _norm(combo.get("supervised_loss")),
    )


# ---- File-based seeding (read finished runs from CSV / pickle) ---------------------


def _row_value(row: pd.Series, col: str) -> Any:
    if col not in row.index:
        return None
    v = row[col]
    return None if pd.isna(v) else v


def load_seed_file(path: Path, dataset_full: str = DATASET) -> set[tuple[Any, ...]]:
    """Load finished combos from a .pkl or .csv file with the inspect_al_runs cache schema.

    Required columns: ``method``, ``strategy``, ``seed``, ``state``, ``dataset``.
    Optional: ``notion``, ``calibration``, ``supervised_loss`` (default to None).
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


# ---- Live wandb query for finished runs --------------------------------------------


def fetch_finished_keys(sources: Iterable[str], dataset_full: str = DATASET) -> set[tuple[Any, ...]]:
    api = wandb.Api()
    keys: set[tuple[Any, ...]] = set()
    for src in sources:
        print(f"Querying wandb {src} for finished {dataset_full} runs...")
        runs = api.runs(src, filters={"state": "finished"}, per_page=500)
        n_seen = 0
        n_match = 0
        for run in runs:
            n_seen += 1
            cfg = run.config or {}
            ds = (cfg.get("dataset") or {}).get("name")
            ds_id = (cfg.get("dataset") or {}).get("openml_id")
            ds_full = f"openml_{ds_id}" if ds == "openml" and ds_id is not None else ds
            if ds_full != dataset_full:
                continue
            method = (cfg.get("method") or {}).get("name")
            strat = (cfg.get("al_strategy") or {}).get("name")
            notion = (cfg.get("al_strategy") or {}).get("notion") if strat == "uncertainty" else None
            seed = cfg.get("seed")
            calib = (cfg.get("calibration") or {}).get("name")
            sup = (cfg.get("supervised_loss") or {}).get("name")
            if method is None or strat is None or seed is None:
                continue
            keys.add(
                _key(
                    {
                        "method": method,
                        "strategy": strat,
                        "notion": notion,
                        "seed": seed,
                        "calibration": calib,
                        "supervised_loss": sup,
                    }
                )
            )
            n_match += 1
        print(f"  [{src}] scanned {n_seen} finished runs, matched {n_match} on {dataset_full}")
    return keys


# ---- Block definitions (cartesian products, exactly as the bash) -------------------


def block_combos() -> Iterator[tuple[str, list[Combo]]]:
    yield (
        "Baselines (base, 3 strategies, 10 seeds)",
        [{"method": "base", "strategy": s, "seed": seed} for s in BASE_STRATEGIES for seed in SEEDS],
    )
    yield (
        "Calibration (base + temp/vector scaling)",
        [
            {"method": "base", "strategy": s, "seed": seed, "calibration": cal}
            for cal in ("temperature_scaling", "vector_scaling")
            for s in BASE_STRATEGIES
            for seed in SEEDS
        ],
    )
    yield (
        "Supervised Loss (base + label_smoothing/relaxation)",
        [
            {"method": "base", "strategy": s, "seed": seed, "supervised_loss": sup}
            for sup in ("label_smoothing", "label_relaxation")
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


def main(argv: Iterable[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        action="append",
        metavar="ENTITY/PROJECT",
        help=(
            "wandb source(s) whose finished runs count as 'done'. Pass multiple times. "
            f"Defaults: {', '.join(DEFAULT_FINISHED_SOURCES)}"
        ),
    )
    p.add_argument(
        "--seed-file",
        action="append",
        metavar="PATH",
        help=(
            "additional .pkl or .csv with finished AL runs (e.g. inspect_al_runs.py's "
            "wandb_cache_runs.pkl) — its finished combos are unioned into the 'done' set. "
            "Repeatable."
        ),
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="skip live wandb query; rely entirely on --seed-file inputs.",
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
        help="when --execute, run at most N missing combos and stop (useful to test).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    sources = args.source or list(DEFAULT_FINISHED_SOURCES)
    for s in sources:
        if s.count("/") != 1 or not all(s.split("/")):
            p.error(f"--source must be ENTITY/PROJECT, got {s!r}")

    seed_files = [Path(p_).expanduser() for p_ in (args.seed_file or [])]
    for sf in seed_files:
        if not sf.exists():
            p.error(f"--seed-file path does not exist: {sf}")

    finished: set[tuple[Any, ...]] = set()
    if not args.no_wandb:
        finished |= fetch_finished_keys(sources)
    elif not seed_files:
        p.error("--no-wandb requires at least one --seed-file")

    if seed_files:
        print(f"Loading {len(seed_files)} seed file(s)...")
        for sf in seed_files:
            finished |= load_seed_file(sf)

    print(f"Total finished {DATASET} combos across all sources: {len(finished)}\n")

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
