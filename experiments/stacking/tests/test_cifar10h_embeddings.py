"""Tests for the cifar10h_embeddings loader using synthetic caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stacking.data import Dataset
from stacking.datasets.cifar10h_embeddings import CIFAR10_CLASS_NAMES, load


def _write_synthetic_caches(root: Path, encoder: str, n_train: int, n_test: int, dim: int) -> None:
    """Write tiny train + test cache .npz files matching the production schema."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"cifar10_{encoder}_train.npz",
        X=rng.standard_normal((n_train, dim)).astype(np.float32),
        y_hard=rng.integers(0, 10, size=n_train).astype(np.int64),
    )
    counts = rng.integers(1, 51, size=(n_test, 10)).astype(np.float32)
    y_soft = counts / counts.sum(axis=1, keepdims=True)
    y_hard = y_soft.argmax(axis=1).astype(np.int64)
    np.savez(
        root / f"cifar10h_{encoder}_test.npz",
        X=rng.standard_normal((n_test, dim)).astype(np.float32),
        y_hard=y_hard,
        y_soft=y_soft,
        counts=counts,
    )


def test_cifar10h_load_returns_dataset(tmp_path: Path) -> None:
    """Loader produces a Dataset whose splits sum to the test cache size."""
    encoder = "siglip2"
    _write_synthetic_caches(tmp_path, encoder, n_train=64, n_test=40, dim=8)
    ds = load(encoder=encoder, calib_frac=0.25, seed=0, cache_root=tmp_path)
    assert isinstance(ds, Dataset)
    assert ds.X_train.shape == (64, 8)
    n_calib = ds.X_calib.shape[0]
    n_test = ds.X_test.shape[0]
    assert n_calib + n_test == 40
    # 0.25 * 40 = 10 calib, 30 test
    assert n_calib == 10
    assert n_test == 30
    assert ds.in_features == 8
    assert ds.num_classes == 10


def test_cifar10h_meta_carries_soft_labels_and_counts(tmp_path: Path) -> None:
    """Meta exposes soft labels and counts partitioned along the same indices."""
    encoder = "siglip2"
    _write_synthetic_caches(tmp_path, encoder, n_train=20, n_test=40, dim=4)
    ds = load(encoder=encoder, calib_frac=0.25, seed=0, cache_root=tmp_path)
    assert ds.meta["name"] == "cifar10h"
    assert ds.meta["encoder"] == encoder
    assert ds.meta["y_soft_calib"].shape == (10, 10)
    assert ds.meta["y_soft_test"].shape == (30, 10)
    assert ds.meta["counts_calib"].shape == (10, 10)
    assert ds.meta["counts_test"].shape == (30, 10)
    assert ds.meta["class_names"] == CIFAR10_CLASS_NAMES


def test_cifar10h_missing_cache_raises_actionable_error(tmp_path: Path) -> None:
    """Missing cache file produces a FileNotFoundError pointing at the cache script."""
    with pytest.raises(FileNotFoundError, match="cache_cifar10h_embeddings"):
        load(encoder="siglip2", cache_root=tmp_path)
