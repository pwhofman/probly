"""Classwise expected calibration error test suite."""

from __future__ import annotations

import pytest

from probly.metrics import classwise_ece


class ClasswiseECESuite:
    """Test suite for classwise_ece."""

    def test_perfectly_calibrated_is_zero(self, array_fn):
        """Predictions matching the empirical class frequencies give zero error."""
        y_true = array_fn([0, 0, 1, 1])
        y_prob = array_fn([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        result = classwise_ece(y_true, y_prob)
        assert float(result) == pytest.approx(0.0)

    def test_shared_bin_calibrated_is_zero(self, array_fn):
        """Within a bin, the empirical frequency only has to match on average."""
        y_true = array_fn([0, 1])
        y_prob = array_fn([[0.5, 0.5], [0.5, 0.5]])
        result = classwise_ece(y_true, y_prob)
        assert float(result) == pytest.approx(0.0)

    def test_known_miscalibration_value(self, array_fn):
        """Constant overconfident predictions give a hand-computable error."""
        # Class 0: both samples predict 0.8 but its empirical frequency is 0.5,
        # so its per-class error is 0.3; class 1 mirrors it. Average: 0.3.
        y_true = array_fn([0, 1])
        y_prob = array_fn([[0.8, 0.2], [0.8, 0.2]])
        result = classwise_ece(y_true, y_prob)
        assert float(result) == pytest.approx(0.3)

    def test_returns_backend_type(self, array_fn, array_type):
        """Result is an instance of the input backend's type."""
        y_true = array_fn([0, 1])
        y_prob = array_fn([[0.9, 0.1], [0.4, 0.6]])
        result = classwise_ece(y_true, y_prob)
        assert isinstance(result, array_type)

    def test_rejects_non_matrix_probabilities(self, array_fn):
        """Probabilities must have shape (n, k)."""
        y_true = array_fn([0, 1])
        y_prob = array_fn([0.5, 0.5])
        with pytest.raises(ValueError, match="shape"):
            classwise_ece(y_true, y_prob)

    def test_rejects_mismatched_labels(self, array_fn):
        """The number of labels must match the number of probability rows."""
        y_true = array_fn([0, 1, 0])
        y_prob = array_fn([[0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="batch size"):
            classwise_ece(y_true, y_prob)
