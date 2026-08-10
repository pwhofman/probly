"""Accuracy test suite."""

from __future__ import annotations

import pytest

from probly.metrics import accuracy


class AccuracySuite:
    """Test suite for accuracy."""

    def test_perfect_predictions(self, array_fn):
        """Matching label predictions give accuracy one."""
        y_true = array_fn([0, 1, 2, 1])
        y_pred = array_fn([0, 1, 2, 1])
        result = accuracy(y_pred, y_true)
        assert float(result) == pytest.approx(1.0)

    def test_known_value(self, array_fn):
        """Half-correct label predictions give accuracy one half."""
        y_true = array_fn([0, 1, 1, 0])
        y_pred = array_fn([0, 1, 0, 1])
        result = accuracy(y_pred, y_true)
        assert float(result) == pytest.approx(0.5)

    def test_probability_matrix_uses_argmax(self, array_fn):
        """A probability matrix is reduced to labels via argmax."""
        y_true = array_fn([0, 1, 1])
        y_prob = array_fn([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        result = accuracy(y_prob, y_true)
        assert float(result) == pytest.approx(2 / 3)

    def test_returns_backend_type(self, array_fn, array_type):
        """Result is an instance of the input backend's type."""
        y_true = array_fn([0, 1])
        y_pred = array_fn([0, 0])
        result = accuracy(y_pred, y_true)
        assert isinstance(result, array_type)

    def test_rejects_bad_ndim(self, array_fn):
        """Predictions must have shape (n,) or (n, k)."""
        y_true = array_fn([0, 1])
        y_pred = array_fn([[[0.9, 0.1]], [[0.2, 0.8]]])
        with pytest.raises(ValueError, match="shape"):
            accuracy(y_pred, y_true)

    def test_rejects_mismatched_labels(self, array_fn):
        """The number of labels must match the number of predictions."""
        y_true = array_fn([0, 1, 0])
        y_pred = array_fn([0, 1])
        with pytest.raises(ValueError, match="batch size"):
            accuracy(y_pred, y_true)
