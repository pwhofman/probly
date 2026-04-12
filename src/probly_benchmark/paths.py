"""Paths for the experiments."""

from __future__ import annotations

from pathlib import Path

DATA_PATH = Path("~/datasets").expanduser()
IMAGENET_SHARD_PATH = Path("~/sharded_imagenet").expanduser()
CHECKPOINT_PATH = Path("~/checkpoints").expanduser()
