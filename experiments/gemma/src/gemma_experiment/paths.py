"""Central path definitions for the gemma experiment."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "model_cache"
RESULTS_DIR = DATA_DIR / "results" / "experiments" / "gemma"
