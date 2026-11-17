"""Where the experiment reads and writes its files."""

from __future__ import annotations

import os
from pathlib import Path

from probly_benchmark.paths import DATA_PATH

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
ASSET_DIR = EXPERIMENT_DIR / "assets"
RESULT_DIR = EXPERIMENT_DIR / "results"
DATA_DIR = Path(os.environ.get("UQXAI_DATA_DIR", DATA_PATH / "uqxai")).expanduser()
"""Downloaded images and cached features (large, not in git). Override with ``UQXAI_DATA_DIR``."""

RIBEIRO_IMAGE = ASSET_DIR / "dog_guitar.jpg"
CASES_MANIFEST = ASSET_DIR / "cases.json"
"""The test images of the uncertainty cases, written by ``scripts/07_select_cases.py``."""


def heads_checkpoint(problem: str) -> Path:
    """Return the file with the trained heads of a problem's sub-ensemble."""
    return RESULT_DIR / problem / "subensemble_heads.pt"


def run_dir(problem: str, image: Path, patch_size: int, baseline: str) -> Path:
    """Return the result directory of one explanation of a problem's sub-ensemble."""
    return RESULT_DIR / problem / f"{image.stem}_patch{patch_size}_{baseline}"
