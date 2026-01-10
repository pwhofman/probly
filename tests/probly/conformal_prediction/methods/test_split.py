"""Tests for Split Conformal method."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from probly.conformal_prediction.methods.split import SplitConformal, SplitConformalClassifier
from probly.conformal_prediction.scores.aps.common import APSScore
from probly.conformal_prediction.scores.lac.common import LACScore


def test_split_conformal_basic() -> None:
    """Test basic split functionality."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 10))
    y = rng.integers(0, 3, 100)

    splitter = SplitConformal(calibration_ratio=0.3, random_state=42)
    x_train, y_train, x_cal, y_cal = splitter.split(x, y)

    assert len(x_train) == 70
    assert len(x_cal) == 30
    assert x_train.shape[1] == 10
    assert len(y_train) == 70
    assert len(y_cal) == 30


def test_split_reproducibility() -> None:
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


def test_split_validation_checks() -> None:
    """Test input validation."""
    splitter = SplitConformal()
    rng = np.random.default_rng(42)

    # Test ratio validation
    x = rng.standard_normal((10, 3))
    y = rng.integers(0, 2, 10)

    with pytest.raises(ValueError, match="calibration_ratio must be in"):
        splitter.split(x, y, calibration_ratio=1.5)

    with pytest.raises(ValueError, match="calibration_ratio must be in"):
        splitter.split(x, y, calibration_ratio=0.0)

    # Test min samples
    x_small = rng.standard_normal((1, 3))
    y_small = rng.integers(0, 2, 1)

    with pytest.raises(ValueError, match="Need at least 2 samples"):
        splitter.split(x_small, y_small)

    # Test length mismatch
    x_mismatch = rng.standard_normal((5, 3))
    y_mismatch = rng.integers(0, 2, 3)

    with pytest.raises(ValueError, match="x and y must have the same length"):
        splitter.split(x_mismatch, y_mismatch)


def test_split_conformal_classifier_initialization() -> None:
    """Test SplitConformalClassifier initialization."""

    class MockModel:
        def __call__(self, x: Sequence[Any]) -> np.ndarray:
            return np.zeros((len(x), 3))

    model = MockModel()

    # Test with APSScore
    aps_score = APSScore(model)
    predictor_aps = SplitConformalClassifier(model, aps_score)
    assert predictor_aps.score is aps_score
    assert predictor_aps.model is model

    # Test with LACScore
    lac_score = LACScore(model)
    predictor_lac = SplitConformalClassifier(model, lac_score, use_accretive=True)
    assert predictor_lac.score is lac_score
    assert predictor_lac.use_accretive is True
