"""Paths for the experiments."""

from __future__ import annotations

import os
from pathlib import Path

DATA_PATH = Path("~/datasets").expanduser()
IMAGENET_SHARD_PATH = Path("~/sharded_imagenet").expanduser()
IMAGENET_TORCH_PATH = Path("~/torch_imagenet").expanduser()
CHECKPOINT_PATH = Path("~/checkpoints").expanduser()
FIGURE_PATH = Path("~/figures").expanduser()


def _default_cache_path() -> Path:
    """Resolve the default wandb cache path.

    Honors the ``PROBLY_CACHE_PATH`` environment variable when set; otherwise
    points at ``probly_cache/`` at the repository root.
    """
    env = os.environ.get("PROBLY_CACHE_PATH")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "probly_cache"


CACHE_PATH = _default_cache_path()
