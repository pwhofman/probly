"""One-time CIFAR-10 + CIFAR-10-H embedding cache builder.

Pulls CIFAR-10 (train) and CIFAR-10-H (test, with human soft labels) via
torchvision and probly's loader, downloads the cifar10h-counts.npy file
from GitHub on first run, then runs the chosen encoder over both splits
and writes two .npz cache files consumed by
``stacking.datasets.cifar10h_embeddings.load``.

Idempotent: skips encoding for any cache file that already exists unless
``--force`` is passed. The counts download is also idempotent.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Final

import numpy as np
import torchvision
from PIL import Image

from probly.datasets.torch import CIFAR10H

from stacking.embed import ENCODERS, cache_path
from stacking.utils import get_device, set_seed

COUNTS_URL: Final[str] = (
    "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy"
)
COUNTS_RELATIVE_PATH: Final[str] = "cifar-10h-master/data/cifar10h-counts.npy"


def _ensure_counts_file(root: Path) -> Path:
    """Download cifar10h-counts.npy under ``<root>/cifar-10h-master/data/`` if missing."""
    target = root / COUNTS_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"Downloading {COUNTS_URL} -> {target}")
        urllib.request.urlretrieve(COUNTS_URL, target)  # noqa: S310
    return target


def _iter_pil_images_cifar10(ds: torchvision.datasets.CIFAR10) -> list[Image.Image]:
    """Materialise CIFAR-10 PIL images. The 50k/10k arrays fit in RAM."""
    images: list[Image.Image] = []
    for i in range(len(ds)):
        arr = ds.data[i]
        images.append(Image.fromarray(arr).convert("RGB"))
    return images


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed-no-compression npz with named arrays."""
    np.savez(path, **arrays)
    summary = ", ".join(f"{k}={v.shape}" for k, v in arrays.items())
    print(f"Wrote {path} ({summary})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        choices=sorted(ENCODERS.keys()),
        required=True,
        help="Encoder to use for both splits.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "cache",
        help="Cache root (also used as the torchvision dataset root).",
    )
    parser.add_argument("--force", action="store_true", help="Re-encode even if cache exists.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the cache builder end-to-end."""
    args = _parse_args()
    set_seed(args.seed)
    args.root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {get_device('auto')}")

    _ensure_counts_file(args.root)

    cifar10_train = torchvision.datasets.CIFAR10(
        root=str(args.root),
        train=True,
        download=True,
        transform=None,
    )
    cifar10h = CIFAR10H(root=str(args.root), transform=None, download=True)

    encoder = ENCODERS[args.encoder]()

    train_path = cache_path(encoder=args.encoder, dataset="cifar10", split="train", root=args.root)
    if args.force or not train_path.exists():
        print(f"Encoding CIFAR-10 train ({len(cifar10_train)} images) -> {train_path}")
        train_images = _iter_pil_images_cifar10(cifar10_train)
        x_train = encoder(train_images)
        y_train = np.asarray(cifar10_train.targets, dtype=np.int64)
        _write_npz(train_path, X=x_train, y_hard=y_train)
    else:
        print(f"Skipping {train_path} (exists; pass --force to overwrite)")

    test_path = cache_path(encoder=args.encoder, dataset="cifar10h", split="test", root=args.root)
    if args.force or not test_path.exists():
        print(f"Encoding CIFAR-10-H test ({len(cifar10h)} images) -> {test_path}")
        test_images = _iter_pil_images_cifar10(cifar10h)
        x_test = encoder(test_images)
        # CIFAR10H reassigns ``self.targets`` to the (N, 10) normalised soft-label tensor
        # after CIFAR10's ``__init__``; ``self.counts`` holds the raw vote counts. Hard
        # labels are the modal vote (argmax of the soft labels), which agrees with the
        # standard CIFAR-10 ground truth on > 99 % of test images.
        y_soft = cifar10h.targets.detach().cpu().numpy().astype(np.float32)
        counts = cifar10h.counts.detach().cpu().numpy().astype(np.float32)
        y_hard = y_soft.argmax(axis=1).astype(np.int64)
        _write_npz(test_path, X=x_test, y_hard=y_hard, y_soft=y_soft, counts=counts)
    else:
        print(f"Skipping {test_path} (exists; pass --force to overwrite)")


if __name__ == "__main__":
    main()
