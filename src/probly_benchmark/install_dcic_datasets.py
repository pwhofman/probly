"""Download and extract DCIC benchmark datasets from Zenodo."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from urllib import request
import zipfile

BASE_URL = "https://zenodo.org/records/7180818/files"
ARCHIVES = {
    "Benthic": "Benthic.zip",
    "CIFAR10H": "CIFAR10H.zip",
    "MiceBone": "MiceBone.zip",
    "Pig": "Pig.zip",
    "Plankton": "Plankton.zip",
    "QualityMRI": "QualityMRI.zip",
    "Synthetic": "Synthetic.zip",
    "Treeversity#1": "Treeversity1.zip",  # zip file name differs from dataset name
    "Treeversity#6": "Treeversity6.zip",  # zip file name differs from dataset name
    "Turkey": "Turkey.zip",
}


def install(name: str, archive: str, dest: Path) -> None:
    """Download and extract one dataset into 'dest/name'."""
    target = dest / name
    if (target / "annotations.json").exists():
        print(f"Skipping {name}: already installed")
        return
    url = f"{BASE_URL}/{archive}?download=1"
    print(f"Downloading {url}")
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / archive
        with request.urlopen(url) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive_path) as archive_file:
            archive_file.extractall(tmp_path)
        extracted = next(p.parent for p in tmp_path.rglob("annotations.json"))
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
    print(f"Installed {name} to {target}")


def main():
    """Download and install every DCIC dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "image",
        help="Directory that will contain the dataset folders.",
    )
    args = parser.parse_args()
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name, archive in ARCHIVES.items():
        install(name, archive, dest)


if __name__ == "__main__":
    main()
