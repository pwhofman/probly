"""Orchestrate calibration experiment generation runs.

Chains multiple generation runs (different temperatures) via direct
function calls. After generation, prints instructions for the manual
review step before analysis.

Usage:
    caffeinate -i uv run python gemma/calibration/run_all.py
"""

from __future__ import annotations

from pathlib import Path
import time

from gemma.calibration.generate import generate_main

RESULTS_DIR = Path("data/results")

RUNS = [
    {"temperature": 0.7, "seed": 42},
    {"temperature": 1.0, "seed": 42},
]

NUM_QUESTIONS = 200
NUM_SAMPLES = 10


def result_path(run: dict) -> Path:
    """Build the output path for a run config."""
    t = str(run["temperature"]).replace(".", "")
    return RESULTS_DIR / f"trivia_t{t}_s{run['seed']}.json"


def main() -> None:
    """Run all generation experiments."""
    t0 = time.time()

    result_files = []
    for run in RUNS:
        t_run = time.time()

        print(f"\n{'=' * 60}")
        print(f"Starting: T={run['temperature']}, seed={run['seed']}")
        print(f"{'=' * 60}\n")

        output = generate_main(
            num_questions=NUM_QUESTIONS,
            num_samples=NUM_SAMPLES,
            temperature=run["temperature"],
            seed=run["seed"],
            output=result_path(run),
        )
        result_files.append(output)

        elapsed = time.time() - t_run
        print(f"\nCompleted T={run['temperature']} in {elapsed / 60:.1f} min")

    total = time.time() - t0
    print(f"\nAll generation done in {total / 60:.1f} min")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review the result files and fill in correctness labels")
    for f in result_files:
        print(f"     - {f}")
    print("  2. Run analysis:")
    files_arg = " ".join(str(f) for f in result_files)
    print(f"     uv run python gemma/calibration/analyze.py --results {files_arg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
