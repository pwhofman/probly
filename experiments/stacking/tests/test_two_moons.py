"""Tests for the two_moons loader."""

from __future__ import annotations

import numpy as np

from stacking.data import Dataset, load_dataset
from stacking.datasets.two_moons import load


def test_two_moons_returns_dataset_with_expected_shape() -> None:
    """Three disjoint splits with 2-D features and binary labels."""
    ds = load(n_samples=400, calib_frac=0.2, test_frac=0.2, seed=0)
    assert isinstance(ds, Dataset)
    assert ds.in_features == 2
    assert ds.num_classes == 2
    total = ds.X_train.shape[0] + ds.X_calib.shape[0] + ds.X_test.shape[0]
    assert total == 400
    assert ds.X_train.shape[1] == 2
    assert ds.y_train.dtype.kind in {"i", "u"}
    assert set(np.unique(ds.y_train).tolist()).issubset({0, 1})


def test_two_moons_seed_is_reproducible() -> None:
    """Same seed yields identical splits."""
    a = load(n_samples=200, seed=7)
    b = load(n_samples=200, seed=7)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_test, b.y_test)


def test_two_moons_via_dispatcher() -> None:
    """load_dataset('two_moons') dispatches correctly."""
    ds = load_dataset("two_moons", n_samples=100, seed=0)
    assert ds.in_features == 2
    assert ds.num_classes == 2


def test_two_moons_accepts_and_ignores_unknown_kwargs() -> None:
    """Passing encoder= (used by cifar10h) must not break two_moons."""
    ds = load_dataset("two_moons", n_samples=100, seed=0, encoder="ignored")
    assert ds.in_features == 2
