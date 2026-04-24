"""File that holds metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetMeta:
    """Dataset class."""

    num_classes: int
    input_dim: tuple[int, ...]


_IMAGENET_META = DatasetMeta(num_classes=1000, input_dim=(224, 224, 3))

DATASETS = {
    "cifar10": DatasetMeta(num_classes=10, input_dim=(32, 32, 3)),
    "fashion_mnist": DatasetMeta(num_classes=10, input_dim=(28, 28, 1)),
    "imagenet": _IMAGENET_META,
    "imagenet_shards": _IMAGENET_META,
}
