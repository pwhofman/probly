"""Shared metrics test suite for all backends."""

from __future__ import annotations

import pytest

from probly.evaluation.active_learning.metrics import compute_accuracy, compute_ece


class MetricsSuite:
    """Backend-agnostic metrics tests. Requires fixtures: array_fn, make_one_hot_probs, make_random_probs."""

    def test_accuracy_perfect(self, array_fn):
        y_pred = array_fn([0, 1, 2, 1, 0])
        y_true = array_fn([0, 1, 2, 1, 0])
        assert compute_accuracy(y_pred, y_true) == 1.0

    def test_accuracy_half_correct(self, array_fn):
        y_pred = array_fn([0, 1, 0, 1])
        y_true = array_fn([0, 0, 1, 1])
        assert compute_accuracy(y_pred, y_true) == pytest.approx(0.5)

    def test_ece_perfect_calibration_near_zero(self, make_one_hot_probs, array_fn):
        y_true = array_fn([0, 1, 2, 3, 0, 1, 2, 3] * 12 + [0, 1, 2, 3])
        probs = make_one_hot_probs(y_true, 4)
        ece = compute_ece(probs, y_true)
        assert ece < 0.05

    def test_ece_returns_float_in_unit_interval(self, make_random_probs):
        probs, y_true = make_random_probs(n=50, n_classes=3, seed=1)
        ece = compute_ece(probs, y_true)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0
