"""Aggregate per-seed JSON result files into mean / std summaries.

Walks ``results/`` (or another directory passed via ``--root``), reads
every ``*.json`` file produced by the stack_* scripts, groups them by
``(experiment, encoder, calibration)``, and computes mean and standard
deviation (population, ``ddof=0`` over the available seeds) of the
metric scalars. Writes a single ``aggregated.json`` summary at the
root and prints a compact table to stdout.

Files whose ``experiment`` key matches ``aggregated`` (i.e. previously
written summaries) are skipped, so re-running the aggregator over a
populated tree is idempotent.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

GROUP_KEYS = ("experiment", "encoder", "calibration")
METRIC_KEYS = (
    "test_acc",
    "ece_uncalibrated",
    "ece_calibrated",
    "conformal_coverage",
    "conformal_avg_set_size",
)


def _load_runs(root: Path) -> list[dict[str, Any]]:
    """Read every JSON under ``root`` and return the per-run records."""
    runs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        with path.open() as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            continue
        if data.get("experiment") in {None, "aggregated"}:
            continue
        runs.append(data)
    return runs


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` over ``values``; std is 0 for length-1 lists."""
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def _aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group runs by ``GROUP_KEYS`` and compute mean / std per metric."""
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = tuple(str(run.get(k)) for k in GROUP_KEYS)
        buckets[key].append(run)

    out: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items()):
        per_run_seeds = sorted({run.get("seed") for run in group})
        metrics_summary: dict[str, dict[str, float]] = {}
        for metric in METRIC_KEYS:
            values: list[float] = []
            for run in group:
                metrics = run.get("metrics") or {}
                if metric in metrics and isinstance(metrics[metric], int | float):
                    values.append(float(metrics[metric]))
            if not values:
                continue
            mean, std = _mean_std(values)
            metrics_summary[metric] = {"mean": mean, "std": std, "n": len(values)}
        out.append(
            {
                "experiment": key[0],
                "encoder": key[1],
                "calibration": key[2],
                "n_runs": len(group),
                "seeds": per_run_seeds,
                "metrics": metrics_summary,
            }
        )
    return out


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print one row per (experiment, encoder, calibration) group."""
    header = f"{'experiment':<24} {'encoder':<24} {'calibration':<18} {'n':>3}  {'acc':>15}  {'ECE_uncal':>15}  {'ECE_cal':>15}  {'cov':>15}  {'set_size':>15}"
    print(header)
    print("-" * len(header))
    for row in rows:
        m = row["metrics"]

        def cell(metric: str) -> str:
            entry = m.get(metric)
            if entry is None:
                return "--"
            return f"{entry['mean']:.4f}+-{entry['std']:.4f}"

        print(
            f"{row['experiment']:<24} {row['encoder']:<24} {row['calibration']:<18} {row['n_runs']:>3}  "
            f"{cell('test_acc'):>15}  {cell('ece_uncalibrated'):>15}  {cell('ece_calibrated'):>15}  "
            f"{cell('conformal_coverage'):>15}  {cell('conformal_avg_set_size'):>15}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Directory tree to walk for per-seed JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the aggregated JSON. Defaults to <root>/aggregated.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Walk the results tree and emit a single aggregated summary."""
    args = _parse_args()
    runs = _load_runs(args.root)
    if not runs:
        print(f"no per-run JSON files found under {args.root}")
        return
    rows = _aggregate(runs)
    _print_table(rows)
    out_path = args.out if args.out is not None else args.root / "aggregated.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump({"experiment": "aggregated", "groups": rows}, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote {out_path} ({len(rows)} groups, {len(runs)} runs)")


if __name__ == "__main__":
    main()
