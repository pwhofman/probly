"""Run all three levels of the Phase-1 investigation end-to-end.

Usage::

    cd experiments/river_uncertainty
    uv run python experiments/run_all.py

The three level scripts are standalone - you can also invoke them
individually. This entry point is simply a convenience for reproducing
the full README figure set from scratch.
"""

from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

LEVEL_SCRIPTS: tuple[str, ...] = (
    "01_stream_response.py",
    "02_ensemble_size_ablation.py",
    "03_uncertainty_drift_detector.py",
)


def main() -> None:
    for script in LEVEL_SCRIPTS:
        path = HERE / script
        print(f"\n=========== {script} ===========")
        runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
