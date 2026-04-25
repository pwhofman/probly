"""Run all (method, stream, seed) combinations and emit paper artifacts."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from river_uq.plotting import build_appendix_figure, build_headline_figure
from river_uq.prequential import run_prequential
from river_uq.tables import build_latency_table, latency_table_to_latex

METHODS: Final[list[str]] = ["arf", "deep_ensemble", "mc_dropout"]
DRIFT_STREAMS: Final[list[str]] = ["stagger_drift", "sea_drift"]
APPENDIX_STREAMS: Final[list[str]] = ["agrawal_stationary"]
DEFAULT_SEEDS: Final[int] = 10
DEFAULT_N: Final[int] = 3000
QUICK_SEEDS: Final[int] = 2
QUICK_N: Final[int] = 1000


def _run_one(args: tuple[str, str, int, int]) -> pd.DataFrame:
    method, stream_name, seed, n_steps = args
    return run_prequential(method=method, stream_name=stream_name, seed=seed, n_steps=n_steps)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode: 2 seeds, 1000 steps.")
    parser.add_argument("--serial", action="store_true", help="Run serially (debug).")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    args = parser.parse_args()

    n_seeds = QUICK_SEEDS if args.quick else DEFAULT_SEEDS
    n_steps = QUICK_N if args.quick else DEFAULT_N
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    drift_combos = [
        (m, s, seed, n_steps)
        for m in METHODS
        for s in DRIFT_STREAMS
        for seed in range(n_seeds)
    ]
    app_combos = [
        (m, s, seed, n_steps)
        for m in METHODS
        for s in APPENDIX_STREAMS
        for seed in range(n_seeds)
    ]
    combos = drift_combos + app_combos

    print(f"Running {len(combos)} combinations (seeds={n_seeds}, n_steps={n_steps})")
    t0 = time.time()
    if args.serial:
        results = [_run_one(c) for c in combos]
    else:
        nprocs = max(1, (os.cpu_count() or 2) - 1)
        with mp.get_context("spawn").Pool(nprocs) as pool:
            results = pool.map(_run_one, combos)
    elapsed = time.time() - t0

    df = pd.concat(results, ignore_index=True)
    parquet_path = results_dir / "run_records.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({len(df)} rows, {elapsed:.1f}s)")

    headline = build_headline_figure(df[df["stream"].isin(DRIFT_STREAMS)])
    headline.savefig(results_dir / "headline_figure.pdf", bbox_inches="tight")
    headline.savefig(results_dir / "headline_figure.png", bbox_inches="tight", dpi=180)
    print(f"Wrote {results_dir / 'headline_figure.pdf'}")

    appendix = build_appendix_figure(df[df["stream"].isin(APPENDIX_STREAMS)])
    appendix.savefig(results_dir / "appendix_stationary.pdf", bbox_inches="tight")
    print(f"Wrote {results_dir / 'appendix_stationary.pdf'}")

    table = build_latency_table(df[df["stream"].isin(DRIFT_STREAMS)])
    table.to_csv(results_dir / "latency_table.csv", index=False)
    (results_dir / "latency_table.tex").write_text(latency_table_to_latex(table))
    print(f"Wrote {results_dir / 'latency_table.csv'} and .tex")

    manifest = {
        "git_sha": _git_sha(),
        "n_seeds": n_seeds,
        "n_steps": n_steps,
        "methods": METHODS,
        "streams": DRIFT_STREAMS + APPENDIX_STREAMS,
        "elapsed_seconds": round(elapsed, 1),
        "n_combinations": len(combos),
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {results_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
