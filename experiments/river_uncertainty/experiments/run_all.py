"""Run all experiment levels end-to-end.

Usage::

    cd experiments/river_uncertainty
    uv run python experiments/run_all.py

The level scripts are standalone - you can also invoke them individually.
This entry point is simply a convenience for reproducing the full figure
set from scratch.  Levels 4-5 require ``torch`` and are skipped if it is
not installed.
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

TORCH_LEVEL_SCRIPTS: tuple[str, ...] = (
    "04_deep_stream_response.py",
    "05_deep_drift_detector.py",
)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def main() -> None:
    for script in LEVEL_SCRIPTS:
        path = HERE / script
        print(f"\n=========== {script} ===========")
        runpy.run_path(str(path), run_name="__main__")

    if _has_torch():
        for script in TORCH_LEVEL_SCRIPTS:
            path = HERE / script
            print(f"\n=========== {script} ===========")
            runpy.run_path(str(path), run_name="__main__")
    else:
        print("\nSkipping levels 4-5 (torch not installed).")


if __name__ == "__main__":
    main()
