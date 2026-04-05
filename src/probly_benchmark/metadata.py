"""File that holds metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetMeta:
    """Dataset class."""

    num_classes: int
    input_dim: tuple[int, ...]


DATASETS = {"cifar10": DatasetMeta(num_classes=10, input_dim=(32, 32, 3))}
