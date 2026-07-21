"""CIFAR-10-H embedding loader: reads cached encoder outputs from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.model_selection import train_test_split

from stacking.data import Dataset
from stacking.embed import cache_path

CIFAR10_CLASS_NAMES: Final[tuple[str, ...]] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
"""Standard CIFAR-10 class names in label order."""


def _default_cache_root() -> Path:
    """Return the default ``cache/`` dir relative to the experiment root."""
    return Path(__file__).resolve().parents[3] / "cache"


def _require_cache_file(path: Path, encoder: str) -> None:
    """Raise an actionable error if the cache file is missing."""
    if not path.exists():
        msg = (
            f"cache file missing: {path}. "
            f"Build it with: python scripts/cache_cifar10h_embeddings.py "
            f"--encoder {encoder}"
        )
        raise FileNotFoundError(msg)


def load(
    *,
    encoder: str,
    calib_frac: float = 0.25,
    seed: int = 0,
    cache_root: Path | None = None,
    **_: Any,  # noqa: ANN401
) -> Dataset:
    """Load the cached CIFAR-10 train embeddings + the CIFAR-10-H test embeddings.

    The 50k CIFAR-10 train cache forms the train split. The 10k
    CIFAR-10-H test cache (with human soft labels) is split stratified on
    ``y_hard`` into calib and test; soft labels and vote counts are
    partitioned along the same indices and stored on ``Dataset.meta``.

    Args:
        encoder: Encoder name; must match the encoder used to write the
            caches (e.g. ``"siglip2"``).
        calib_frac: Fraction of the 10k CIFAR-10-H samples reserved for
            calibration; the rest forms the test split. Default ``0.25``
            gives 2 500 calib / 7 500 test.
        seed: Seed for the calib/test split.
        cache_root: Directory holding the cache files. Defaults to
            ``experiments/stacking/cache/``.
        **_: Loader-agnostic kwargs (e.g. ``n_samples=`` for ``two_moons``)
            are accepted and ignored.

    Returns:
        A :class:`stacking.data.Dataset` with ``num_classes=10`` and
        ``in_features`` derived from the cached embedding dimension.

    Raises:
        FileNotFoundError: If either cache file is missing; the error
            message includes the exact command needed to build it.
    """
    root = cache_root if cache_root is not None else _default_cache_root()
    train_path = cache_path(encoder=encoder, dataset="cifar10", split="train", root=root)
    test_path = cache_path(encoder=encoder, dataset="cifar10h", split="test", root=root)
    _require_cache_file(train_path, encoder)
    _require_cache_file(test_path, encoder)

    train_npz = np.load(train_path)
    test_npz = np.load(test_path)

    X_train = train_npz["X"].astype(np.float32)
    y_train = train_npz["y_hard"].astype(np.int64)

    X_pool = test_npz["X"].astype(np.float32)
    y_pool_hard = test_npz["y_hard"].astype(np.int64)
    y_pool_soft = test_npz["y_soft"].astype(np.float32)
    counts_pool = test_npz["counts"].astype(np.float32)

    indices = np.arange(X_pool.shape[0])
    idx_calib, idx_test = train_test_split(
        indices, test_size=1.0 - calib_frac, stratify=y_pool_hard, random_state=seed,
    )

    in_features = int(X_train.shape[1])

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_calib=X_pool[idx_calib],
        y_calib=y_pool_hard[idx_calib],
        X_test=X_pool[idx_test],
        y_test=y_pool_hard[idx_test],
        in_features=in_features,
        num_classes=10,
        meta={
            "name": "cifar10h",
            "encoder": encoder,
            "seed": seed,
            "y_soft_calib": y_pool_soft[idx_calib],
            "y_soft_test": y_pool_soft[idx_test],
            "counts_calib": counts_pool[idx_calib],
            "counts_test": counts_pool[idx_test],
            "class_names": CIFAR10_CLASS_NAMES,
        },
    )
