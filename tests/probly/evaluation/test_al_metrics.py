"""Tests for the active learning evaluation metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from probly.evaluation.active_learning.metrics import (  # noqa: E402
    compute_accuracy,
    compute_ece,
    compute_nauc,
)

# ---------------------------------------------------------------------------
# compute_accuracy
# ---------------------------------------------------------------------------


def test_accuracy_perfect():
    y_pred = np.array([0, 1, 2, 1, 0])
    y_true = np.array([0, 1, 2, 1, 0])
    assert compute_accuracy(y_pred, y_true) == 1.0


def test_accuracy_half_correct():
    y_pred = np.array([0, 1, 0, 1])
    y_true = np.array([0, 0, 1, 1])
    assert compute_accuracy(y_pred, y_true) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------


def _make_one_hot_probs(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """Build one-hot probability matrix matching labels exactly."""
    probs = np.zeros((len(y_true), n_classes), dtype=np.float32)
    probs[np.arange(len(y_true)), y_true] = 1.0
    return probs


def test_ece_perfect_calibration_near_zero():
    rng = np.random.default_rng(0)
    n = 100
    n_classes = 4
    y_true = rng.integers(0, n_classes, size=n)
    probs = _make_one_hot_probs(y_true, n_classes)
    ece = compute_ece(probs, y_true)
    assert ece < 0.05


def test_ece_returns_float_in_unit_interval():
    rng = np.random.default_rng(1)
    n = 50
    n_classes = 3
    y_true = rng.integers(0, n_classes, size=n)
    raw = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float32)
    ece = compute_ece(raw, y_true)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# compute_nauc
# ---------------------------------------------------------------------------


def test_nauc_constant_one():
    assert compute_nauc([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_nauc_constant_below_one():
    assert compute_nauc([0.8, 0.8, 0.8, 0.8]) == pytest.approx(0.8)


def test_nauc_fast_improver_beats_slow():
    slow = compute_nauc([0.5, 0.5, 0.5, 0.8])
    fast = compute_nauc([0.5, 0.6, 0.7, 0.8])
    assert slow < fast


def test_nauc_result_in_unit_interval():
    nauc = compute_nauc([0.3, 0.5, 0.7, 0.9])
    assert 0.0 <= nauc <= 1.0


def test_nauc_single_value_is_nan():
    result = compute_nauc([0.7])
    assert math.isnan(result)
