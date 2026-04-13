"""Orchestrate the full calibration experiment pipeline.

Chains multiple experiment runs (different temperatures) and generates
plots from all results at the end.

Usage:
    caffeinate -i uv run python gemma/calibration/run_all.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

RESULTS_DIR = Path("data/results")
FIGURES_DIR = Path("data/figures")

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


def run_experiment(run: dict) -> Path:
    """Run a single experiment configuration."""
    output = result_path(run)
    cmd = [
        sys.executable,
        "gemma/calibration/run_experiment.py",
        "--num-questions",
        str(NUM_QUESTIONS),
        "--num-samples",
        str(NUM_SAMPLES),
        "--temperature",
        str(run["temperature"]),
        "--seed",
        str(run["seed"]),
        "--output",
        str(output),
    ]
    print(f"\n{'=' * 60}")
    print(f"Starting: T={run['temperature']}, seed={run['seed']}")
    print(f"Output:   {output}")
    print(f"{'=' * 60}\n")

    subprocess.run(cmd, check=True)  # noqa: S603
    return output


def run_plots(result_files: list[Path]) -> None:
    """Generate plots from all completed result files."""
    cmd = [
        sys.executable,
        "gemma/calibration/plot_calibration.py",
        "--results",
        *[str(f) for f in result_files],
        "--output",
        str(FIGURES_DIR),
    ]
    print(f"\n{'=' * 60}")
    print("Generating figures")
    print(f"{'=' * 60}\n")

    subprocess.run(cmd, check=True)  # noqa: S603


def main() -> None:
    """Run all experiments and generate plots."""
    t0 = time.time()

    result_files = []
    for run in RUNS:
        t_run = time.time()
        output = run_experiment(run)
        elapsed = time.time() - t_run
        print(f"\nCompleted T={run['temperature']} in {elapsed / 60:.1f} min")
        result_files.append(output)

    run_plots(result_files)

    total = time.time() - t0
    print(f"\nAll done in {total / 60:.1f} min")


if __name__ == "__main__":
    main()
