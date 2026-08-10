"""Confidence expected calibration error test suite."""

from __future__ import annotations

import pytest

from probly.metrics import expected_calibration_error


class ExpectedCalibrationErrorSuite:
    """Test suite for expected_calibration_error."""

    def test_confident_correct_is_zero(self, array_fn):
        """Fully confident and correct predictions give zero error."""
        y_true = array_fn([0, 1])
        y_prob = array_fn([[1.0, 0.0], [0.0, 1.0]])
        result = expected_calibration_error(y_prob, y_true)
        assert float(result) == pytest.approx(0.0)

    def test_confidence_matching_accuracy_is_zero(self, array_fn):
        """Within a bin, the accuracy only has to match the mean confidence."""
        # All four samples share confidence 0.75 (one bin); three are correct.
        y_true = array_fn([0, 0, 0, 1])
        y_prob = array_fn([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]])
        result = expected_calibration_error(y_prob, y_true)
        assert float(result) == pytest.approx(0.0)

    def test_known_miscalibration_value(self, array_fn):
        """Constant overconfident predictions give a hand-computable error."""
        # Both samples predict class 0 with confidence 0.8 but only one is
        # correct, so the single occupied bin contributes |0.5 - 0.8| = 0.3.
        y_true = array_fn([0, 1])
        y_prob = array_fn([[0.8, 0.2], [0.8, 0.2]])
        result = expected_calibration_error(y_prob, y_true)
        assert float(result) == pytest.approx(0.3)

    def test_returns_backend_type(self, array_fn, array_type):
        """Result is an instance of the input backend's type."""
        y_true = array_fn([0, 1])
        y_prob = array_fn([[0.9, 0.1], [0.4, 0.6]])
        result = expected_calibration_error(y_prob, y_true)
        assert isinstance(result, array_type)

    def test_rejects_non_matrix_probabilities(self, array_fn):
        """Probabilities must have shape (n, k)."""
        y_true = array_fn([0, 1])
        y_prob = array_fn([0.5, 0.5])
        with pytest.raises(ValueError, match="shape"):
            expected_calibration_error(y_prob, y_true)

    def test_rejects_mismatched_labels(self, array_fn):
        """The number of labels must match the number of probability rows."""
        y_true = array_fn([0, 1, 0])
        y_prob = array_fn([[0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="batch size"):
            expected_calibration_error(y_prob, y_true)

    @pytest.mark.parametrize("num_bins", [0, -1])
    def test_rejects_non_positive_num_bins(self, array_fn, num_bins):
        """num_bins must be at least one."""
        y_true = array_fn([0, 1])
        y_prob = array_fn([[0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="num_bins"):
            expected_calibration_error(y_prob, y_true, num_bins=num_bins)
