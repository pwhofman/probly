"""Downloads single ImageNet-1k classes from a public parquet mirror of the dataset.

The mirror ``benjamin-paine/imagenet-1k-256x256`` stores the images in row groups of 100 with
statistics of their labels. Only the row groups that can hold a wanted class are read: the training
images are sorted by label, so a class costs about 20 megabytes of download.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from huggingface_hub import HfFileSystem
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

IMAGENET_MIRROR = "datasets/benjamin-paine/imagenet-1k-256x256"


def class_file(directory: Path, split: str, imagenet_class: int) -> Path:
    """Return the parquet file with the images of one ImageNet class of a split."""
    return directory / split / f"{imagenet_class}.parquet"


def download_classes(split: str, imagenet_classes: Iterable[int], directory: Path) -> None:
    """Download the images of ImageNet classes that are not on disk yet, one parquet file per class.

    Args:
        split: ``"train"`` or ``"validation"``.
        imagenet_classes: ImageNet-1k class indices.
        directory: Directory of the files; see :func:`class_file`. Each file has one column
            ``image`` with the encoded bytes of the images.
    """
    wanted = np.array(sorted({c for c in imagenet_classes if not class_file(directory, split, c).exists()}))
    if not len(wanted):
        return
    filesystem = HfFileSystem()
    images: dict[int, list[bytes]] = {int(imagenet_class): [] for imagenet_class in wanted}
    for shard in tqdm(sorted(filesystem.glob(f"{IMAGENET_MIRROR}/data/{split}-*.parquet")), desc=f"{split} shards"):
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
        for imagenet_class in np.intersect1d(labels, wanted):
            rows = pa.array(np.flatnonzero(labels == imagenet_class))
            images[int(imagenet_class)].extend(image_bytes.take(rows).to_pylist())
    for imagenet_class, class_images in images.items():
        path = class_file(directory, split, imagenet_class)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"image": pa.array(class_images, type=pa.binary())}), path)
