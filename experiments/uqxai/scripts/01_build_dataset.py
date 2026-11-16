"""Download the three-class ImageNet subset (guitar, dog, human).

The groups are balanced by image count: every guitar and human class is kept in full, and the 118 dog
breeds are subsampled to the same order of magnitude.
"""

from __future__ import annotations

import argparse

import numpy as np
from uqxai.classes import CLASS_NAMES
from uqxai.data import SubsetSpec, build_subset
from uqxai.paths import DATA_DIR

SPECS = {
    "train": SubsetSpec("train", {"guitar": None, "dog": 32, "human": None}),
    "validation": SubsetSpec("validation", {"guitar": None, "dog": 3, "human": None}),
}


def main() -> None:
    """Build the train and validation subsets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name, spec in SPECS.items():
        path = DATA_DIR / f"{name}.parquet"
        if path.exists():
            print(f"{path} exists, skipping")
            continue
        table = build_subset(spec, path, seed=args.seed)
        counts = np.bincount(table.column("label").to_numpy(), minlength=len(CLASS_NAMES))
        print(f"{name}: {dict(zip(CLASS_NAMES, counts.tolist(), strict=True))} -> {path}")


if __name__ == "__main__":
    main()
