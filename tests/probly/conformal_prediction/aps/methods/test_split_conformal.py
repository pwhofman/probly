"""Tests for Split Conformal APS method."""

from __future__ import annotations

import numpy as np
import pytest

from probly.conformal_prediction.aps.methods.split_conformal import SplitConformal


def test_basic_split() -> None:
    """Test that split returns correct shapes."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 10))
    y = rng.integers(0, 3, 100)

    splitter = SplitConformal(calibration_ratio=0.3, random_state=42)
    x_train, y_train, x_cal, y_cal = splitter.split(x, y)

    assert len(x_train) == 70
    assert len(x_cal) == 30
    assert x_train.shape[1] == 10


def test_reproducibility() -> None:
    """Test that same random_state gives same results."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal((50, 5))
    y = rng.integers(0, 2, 50)

    splitter1 = SplitConformal(random_state=42)
    splitter2 = SplitConformal(random_state=42)

    x_train1, y_train1, x_cal1, y_cal1 = splitter1.split(x, y)
    x_train2, y_train2, x_cal2, y_cal2 = splitter2.split(x, y)

    assert np.array_equal(x_train1, x_train2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(x_cal1, x_cal2)
    assert np.array_equal(y_cal1, y_cal2)


def test_parameter_override() -> None:
    """Test that calibration_ratio can be overridden."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 3))
    y = rng.integers(0, 3, 100)

    splitter = SplitConformal(calibration_ratio=0.3)

    # Default (30%)
    _, _, x_cal1, _ = splitter.split(x, y)
    # Override (20%)
    _, _, x_cal2, _ = splitter.split(x, y, calibration_ratio=0.2)

    assert len(x_cal1) != len(x_cal2)
    assert len(x_cal1) == 30  # 100 * 0.3
    assert len(x_cal2) == 20  # 100 * 0.2


def test_validation_checks() -> None:
    """Test validation of inputs."""
    rng = np.random.default_rng(42)
    splitter = SplitConformal()

    # Test ratio validation
    x = rng.standard_normal((10, 3))
    y = rng.integers(0, 2, 10)

    with pytest.raises(ValueError, match="calibration_ratio must be between"):
        splitter.split(x, y, calibration_ratio=1.5)

    with pytest.raises(ValueError, match="calibration_ratio must be between"):
        splitter.split(x, y, calibration_ratio=0.0)

    # Test min samples
    x_small = rng.standard_normal((1, 3))
    y_small = rng.integers(0, 2, 1)

    with pytest.raises(ValueError, match="Need at least 2 samples"):
        splitter.split(x_small, y_small)

    # Test length mismatch
    x_mismatch = rng.standard_normal((5, 3))
    y_mismatch = rng.integers(0, 2, 3)

    with pytest.raises(ValueError, match="x and y must have same length"):
        splitter.split(x_mismatch, y_mismatch)


def test_get_split_info() -> None:
    """Test get_split_info method."""
    rng = np.random.default_rng(42)
    x: np.ndarray = rng.standard_normal((20, 3))
    y: np.ndarray = rng.integers(0, 2, 20)

    splitter = SplitConformal()

    # Before split
    info_before = splitter.get_split_info()
    assert isinstance(info_before, dict)
    assert "status" in info_before

    # After split
    splitter.split(x, y)
    info_after = splitter.get_split_info()

    # check type: should be SplitInfo
    assert isinstance(info_after, dict)
    assert "n_training" in info_after
    assert "n_calibration" in info_after

    # type shoud be int
    n_train = info_after["n_training"]
    n_cal = info_after["n_calibration"]

    # explicit type checks
    assert isinstance(n_train, int), f"Expected int, got {type(n_train)}"
    assert isinstance(n_cal, int), f"Expected int, got {type(n_cal)}"

    # check sums to total samples
    total = n_train + n_cal
    assert total == len(x)


def test_string_representation() -> None:
    """Test __str__ method."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((10, 3))
    y = rng.integers(0, 2, 10)

    splitter = SplitConformal(calibration_ratio=0.3, random_state=42)

    # Before split
    str_before = str(splitter)
    assert "SplitConformal" in str_before
    assert "ratio=0.3" in str_before

    # After split
    splitter.split(x, y)
    str_after = str(splitter)
    assert "Training" in str_after
    assert "Calibration" in str_after
