"""Tests for the Dataset dataclass and load_dataset dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from stacking.data import Dataset, load_dataset


def test_dataset_dataclass_holds_arrays_and_dims() -> None:
    """Dataset stores the six arrays and the in_features / num_classes scalars."""
    rng = np.random.default_rng(0)
    ds = Dataset(
        X_train=rng.standard_normal((4, 3)),
        y_train=np.array([0, 1, 0, 1]),
        X_calib=rng.standard_normal((2, 3)),
        y_calib=np.array([0, 1]),
        X_test=rng.standard_normal((2, 3)),
        y_test=np.array([1, 0]),
        in_features=3,
        num_classes=2,
    )
    assert ds.in_features == 3
    assert ds.num_classes == 2
    assert ds.meta == {}
    assert ds.X_train.shape == (4, 3)


def test_dataset_is_frozen() -> None:
    """Dataset is a frozen dataclass: assignment must raise."""
    rng = np.random.default_rng(0)
    ds = Dataset(
        X_train=rng.standard_normal((1, 1)),
        y_train=np.array([0]),
        X_calib=rng.standard_normal((1, 1)),
        y_calib=np.array([0]),
        X_test=rng.standard_normal((1, 1)),
        y_test=np.array([0]),
        in_features=1,
        num_classes=2,
    )
    with pytest.raises((AttributeError, Exception)):
        ds.in_features = 99  # ty: ignore[invalid-assignment]


def test_load_dataset_unknown_name_raises() -> None:
    """Dispatcher raises ValueError on unknown dataset names."""
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("nope")
