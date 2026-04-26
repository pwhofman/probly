"""Run (method x stream x seed) prequential combinations and write a parquet.

Pure runner: no figures, no tables, no manifest. Use ``plot_results.py`` to
turn the parquet into per-stream figures.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Final

import pandas as pd

from river_uq.prequential import run_prequential
from river_uq.streams import STREAM_NAMES

DEFAULT_METHODS: Final[tuple[str, ...]] = ("arf", "deep_ensemble", "mc_dropout")
DEFAULT_SEEDS: Final[int] = 10
DEFAULT_N: Final[int] = 3000
QUICK_SEEDS: Final[int] = 2
QUICK_N: Final[int] = 1000


def _run_one(args: tuple[str, str, int, int]) -> pd.DataFrame:
    method, stream_name, seed, n_steps = args
    return run_prequential(
        method=method, stream_name=stream_name, seed=seed, n_steps=n_steps
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(STREAM_NAMES),
        help=f"Streams to run. Choose from: {', '.join(STREAM_NAMES)}.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help=f"UQ methods to run. Choose from: {', '.join(DEFAULT_METHODS)}.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit seed list (overrides --n-seeds).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help="Number of seeds to run (0..n_seeds-1) if --seeds not given.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N,
        help="Number of stream samples per run.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument(
        "--output-name",
        default="run_records",
        help="Parquet filename stem (no extension).",
    )
    parser.add_argument("--serial", action="store_true", help="Disable multiprocessing.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Quick mode: {QUICK_SEEDS} seeds, {QUICK_N} steps.",
    )
    return parser.parse_args()


def _validate(streams: list[str], methods: list[str]) -> None:
    bad_streams = [s for s in streams if s not in STREAM_NAMES]
    if bad_streams:
        msg = (
            f"unknown stream(s): {bad_streams}. "
            f"Choose from: {', '.join(STREAM_NAMES)}."
        )
        raise SystemExit(msg)
    bad_methods = [m for m in methods if m not in DEFAULT_METHODS]
    if bad_methods:
        msg = (
            f"unknown method(s): {bad_methods}. "
            f"Choose from: {', '.join(DEFAULT_METHODS)}."
        )
        raise SystemExit(msg)


def main() -> None:
    args = _parse_args()
    _validate(args.streams, args.methods)

    if args.quick:
        n_seeds = QUICK_SEEDS
        n_steps = QUICK_N
    else:
        n_seeds = args.n_seeds
        n_steps = args.n_steps
    seeds = args.seeds if args.seeds is not None else list(range(n_seeds))

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = [
        (method, stream, seed, n_steps)
        for method in args.methods
        for stream in args.streams
        for seed in seeds
    ]
    print(
        f"Running {len(combos)} combinations "
        f"({len(args.methods)} methods x {len(args.streams)} streams "
        f"x {len(seeds)} seeds, n_steps={n_steps})"
    )

    t0 = time.time()
    if args.serial:
        results = [_run_one(c) for c in combos]
    else:
        nprocs = max(1, (os.cpu_count() or 2) - 1)
        with mp.get_context("spawn").Pool(nprocs) as pool:
            results = pool.map(_run_one, combos)
    elapsed = time.time() - t0

    df = pd.concat(results, ignore_index=True)
    parquet_path = results_dir / f"{args.output_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({len(df)} rows, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
