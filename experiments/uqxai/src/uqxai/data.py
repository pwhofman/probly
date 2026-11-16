"""Builds the three-class ImageNet subset from a public parquet mirror of ImageNet-1k.

The mirror ``benjamin-paine/imagenet-1k-256x256`` stores the images sorted by label, in row groups of
100 images. Only the row groups of the wanted classes are read, so the subset costs a few hundred
megabytes of download instead of the full dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from huggingface_hub import HfFileSystem
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from uqxai.classes import CLASS_GROUPS, IMAGENET_TO_GROUP

if TYPE_CHECKING:
    from pathlib import Path

IMAGENET_MIRROR = "datasets/benjamin-paine/imagenet-1k-256x256"


@dataclass(frozen=True)
class SubsetSpec:
    """How many images of a split to keep.

    Attributes:
        split: ``"train"`` or ``"validation"``.
        images_per_class: Images kept per ImageNet class, by group name. ``None`` keeps all of them.
    """

    split: str
    images_per_class: dict[str, int | None]


def build_subset(spec: SubsetSpec, path: Path, seed: int = 0) -> pa.Table:
    """Download the images of the three groups and write them to one parquet file.

    Args:
        spec: The split and the number of images per ImageNet class.
        path: Parquet file to write, with the columns ``image`` (encoded bytes), ``imagenet_label``
            and ``label`` (index of the group).
        seed: Seed of the per-class subsampling.

    Returns:
        The written table.
    """
    rng = np.random.default_rng(seed)
    filesystem = HfFileSystem()
    shards = sorted(filesystem.glob(f"{IMAGENET_MIRROR}/data/{spec.split}-*.parquet"))
    wanted = np.array(sorted(IMAGENET_TO_GROUP))
    group_names = list(CLASS_GROUPS)
    images: list[bytes] = []
    imagenet_labels: list[int] = []
    for shard in tqdm(shards, desc=f"{spec.split} shards"):
        with filesystem.open(shard, "rb") as file:
            parquet = pq.ParquetFile(file)
            # Index among the leaf columns: the image struct has two leaves before the label.
            label_column = parquet.schema.names.index("label")
            row_groups = []
            for index in range(parquet.metadata.num_row_groups):
                stats = parquet.metadata.row_group(index).column(label_column).statistics
                if np.any((wanted >= stats.min) & (wanted <= stats.max)):
                    row_groups.append(index)
            if not row_groups:
                continue
            table = parquet.read_row_groups(row_groups, columns=["image", "label"])
        labels = table.column("label").to_numpy()
        image_bytes = table.column("image").combine_chunks().field("bytes")
        for imagenet_label in np.intersect1d(np.unique(labels), wanted):
            rows = np.flatnonzero(labels == imagenet_label)
            limit = spec.images_per_class[group_names[IMAGENET_TO_GROUP[int(imagenet_label)]]]
            if limit is not None and len(rows) > limit:
                rows = np.sort(rng.choice(rows, size=limit, replace=False))
            images.extend(image_bytes.take(pa.array(rows)).to_pylist())
            imagenet_labels.extend([int(imagenet_label)] * len(rows))
    subset = pa.table(
        {
            "image": pa.array(images, type=pa.binary()),
            "imagenet_label": pa.array(imagenet_labels, type=pa.int16()),
            "label": pa.array([IMAGENET_TO_GROUP[label] for label in imagenet_labels], type=pa.int8()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, path)
    return subset
