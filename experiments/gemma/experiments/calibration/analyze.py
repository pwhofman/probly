"""Analyze reviewed calibration results.

Loads result JSON files where correctness labels have been filled in
and computes aggregate calibration metrics. Handles partially-reviewed
files by filtering out unlabeled items.

Usage:
    uv run python experiments/calibration/analyze.py \
        --results data/results/experiments/gemma/run_t07.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_experiment.calibration import compute_aggregates


def load_results(paths: list[str]) -> list[dict]:
    """Load result files and return list of run dicts."""
    runs = []
    for p in paths:
        with Path(p).open() as f:
            runs.append(json.load(f))
    return runs


def analyze_run(run: dict) -> dict:
    """Compute aggregates for a single run and update it in place.

    Args:
        run: A run dict with ``metadata``, ``results``, ``aggregate`` keys.
    """
    all_results = run["results"]
    labeled = [
        r for r in all_results if r.get("is_correct_discrete") is not None and r.get("is_correct_weighted") is not None
    ]
    total = len(all_results)
    n_labeled = len(labeled)
    skipped = total - n_labeled

    if skipped > 0:
        print(f"  Using {n_labeled}/{total} labeled items ({skipped} skipped, correctness=null)")
    else:
        print(f"  All {total} items labeled")

    if not labeled:
        print("  WARNING: No labeled items, skipping aggregate computation")
        return run

    run["aggregate"] = compute_aggregates(labeled)
    run["metadata"]["step"] = "analyzed"
    return run


def analyze_main(
    *,
    results: list[str | Path],
    save_aggregates: bool = True,
) -> None:
    """Load reviewed results and compute calibration metrics.

    Args:
        results: Paths to result JSON files.
        save_aggregates: Whether to write aggregates back into the JSON files.
    """
    result_paths = [str(p) for p in results]

    print(f"Loading {len(result_paths)} result file(s)...")
    runs = load_results(result_paths)

    for i, run in enumerate(runs):
        t = run["metadata"]["temperature"]
        print(f"\n  Run {i + 1}: T={t}")
        analyze_run(run)

        if save_aggregates and run["aggregate"] is not None:
            path = Path(result_paths[i])
            with path.open("w") as f:
                json.dump(run, f, indent=2)
            print(f"  Updated aggregates in {path}")

    # Print summary
    for run, path in zip(runs, result_paths, strict=True):
        agg = run.get("aggregate")
        if agg is None:
            continue
        print(f"\n--- {Path(path).name} ---")
        print(f"Accuracy:       {agg['accuracy']:.1%}")
        print(f"ECE (discrete): {agg['ece_discrete']:.4f}")
        print(f"ACE (discrete): {agg['ace_discrete']:.4f}")

    print("\nDone!")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more reviewed JSON result files.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write aggregates back into the JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run analysis on reviewed calibration results."""
    args = parse_args()
    analyze_main(
        results=args.results,
        save_aggregates=not args.no_save,
    )


if __name__ == "__main__":
    main()
