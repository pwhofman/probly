"""File that holds metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetMeta:
    """Dataset class."""

    num_classes: int
    input_dim: tuple[int, ...]
    base_model: str = ""
    type: str = "image"


_IMAGENET_META = DatasetMeta(num_classes=1000, input_dim=(224, 224, 3))

DATASETS = {
    "cifar10": DatasetMeta(num_classes=10, input_dim=(32, 32, 3), base_model="resnet18"),
    "fashion_mnist": DatasetMeta(num_classes=10, input_dim=(28, 28, 1), base_model="lenet"),
    "imagenet": _IMAGENET_META,
    "imagenet_shards": _IMAGENET_META,
}

# Active learning datasets keyed by Hydra config name (al_dataset/*.yaml).
AL_DATASETS: dict[str, DatasetMeta] = {
    "cifar10": DATASETS["cifar10"],
    "fashion_mnist": DATASETS["fashion_mnist"],
    "openml_6": DatasetMeta(num_classes=26, input_dim=(), base_model="tabular_mlp", type="tabular"),
    "openml_155": DatasetMeta(num_classes=10, input_dim=(), base_model="tabular_mlp", type="tabular"),
    "openml_156": DatasetMeta(num_classes=5, input_dim=(), base_model="tabular_mlp", type="tabular"),
}
