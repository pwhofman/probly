"""Download the ImageNet classes of the problems: all training and validation images of each class."""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq
from uqxai.classes import PROBLEMS
from uqxai.data import class_file, download_classes
from uqxai.paths import DATA_DIR


def main() -> None:
    """Download the classes of the chosen problems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    args = parser.parse_args()
    names = {index: name for problem in args.problems for name, index in PROBLEMS[problem].classes.items()}
    for split in ("train", "validation"):
        download_classes(split, names, DATA_DIR)
        counts = {name: pq.read_metadata(class_file(DATA_DIR, split, index)).num_rows for index, name in names.items()}
        print(f"{split}: {counts}")


if __name__ == "__main__":
    main()
