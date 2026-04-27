"""Orchestrate calibration experiment generation runs.

Chains multiple generation runs (different temperatures) via direct
function calls. After generation, prints instructions for the manual
review step before analysis.

Usage:
    caffeinate -i uv run python experiments/calibration/run_all.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Allow importing sibling modules when run as a script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from generate import generate_main  # noqa: E402

from gemma_experiment import RESULTS_DIR  # noqa: E402

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
    print(f"     uv run python experiments/calibration/analyze.py --results {files_arg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
