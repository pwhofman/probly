"""Script to generate sharded WebDataset from ImageNet."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from tqdm import tqdm
import webdataset as wds

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def main(root: Path, output_dir: Path, split: str, maxcount: int) -> None:
    """Shard an ImageNet split directory into WebDataset tar files."""
    random.seed(1)
    split_dir = root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / f"imagenet-{split}-%06d.tar")

    classes = sorted(p for p in split_dir.iterdir() if p.is_dir())

    all_files: list[tuple[int, str, Path]] = []
    for cls_idx, cls_dir in enumerate(classes):
        cls_name = cls_dir.name
        all_files.extend(
            (cls_idx, cls_name, fpath) for fpath in sorted(cls_dir.iterdir()) if fpath.suffix.lower() in IMAGE_SUFFIXES
        )

    random.shuffle(all_files)

    with wds.ShardWriter(output_pattern, maxcount=maxcount) as sink:  # ty: ignore[unresolved-attribute]
        for cls_idx, cls_name, filepath in tqdm(all_files, desc=f"Images ({split})"):
            image_data = filepath.read_bytes()
            sink.write(
                {
                    "__key__": f"{cls_name}/{filepath.stem}",
                    "jpg": image_data,
                    "txt": str(cls_idx),
                }
            )

        print(f"Done. Wrote {len(all_files)} samples across {sink.shard} shards.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sharded WebDataset from ImageNet.")
    parser.add_argument("--root", type=Path, help="Path to the ImageNet split directory (e.g. /data/imagenet)")
    parser.add_argument("--output_dir", type=Path, help="Output directory for shard tar files")
    parser.add_argument("--split", type=str, required=True, help="Split name (e.g. train, val)")
    parser.add_argument(
        "--maxcount",
        type=int,
        default=1000,
        help="Maximum number of samples per shard (default: 1000)",
    )
    args = parser.parse_args()
    main(root=args.root, output_dir=args.output_dir, split=args.split, maxcount=args.maxcount)
