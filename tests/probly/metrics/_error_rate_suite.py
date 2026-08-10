"""False positive rate and false negative rate test suite."""

from __future__ import annotations

import math

import pytest

from probly.metrics import false_negative_rate, false_positive_rate


class ErrorRateSuite:
    """Test suite for false_positive_rate and false_negative_rate."""

    def test_perfect_predictions_are_zero(self, array_fn):
        """Matching predictions give zero false positive and negative rates."""
        y_true = array_fn([0, 1, 1, 0])
        y_pred = array_fn([0, 1, 1, 0])
        assert float(false_positive_rate(y_pred, y_true)) == pytest.approx(0.0)
        assert float(false_negative_rate(y_pred, y_true)) == pytest.approx(0.0)

    def test_fpr_known_value(self, array_fn):
        """One of two negatives predicted positive gives a rate of one half."""
        y_true = array_fn([0, 0, 1, 1])
        y_pred = array_fn([1, 0, 1, 1])
        result = false_positive_rate(y_pred, y_true)
        assert float(result) == pytest.approx(0.5)

    def test_fnr_known_value(self, array_fn):
        """One of two positives predicted negative gives a rate of one half."""
        y_true = array_fn([0, 0, 1, 1])
        y_pred = array_fn([0, 0, 0, 1])
        result = false_negative_rate(y_pred, y_true)
        assert float(result) == pytest.approx(0.5)

    def test_fpr_without_negatives_is_nan(self, array_fn):
        """The false positive rate is undefined without negative samples."""
        y_true = array_fn([1, 1])
        y_pred = array_fn([1, 0])
        assert math.isnan(float(false_positive_rate(y_pred, y_true)))

    def test_fnr_without_positives_is_nan(self, array_fn):
        """The false negative rate is undefined without positive samples."""
        y_true = array_fn([0, 0])
        y_pred = array_fn([1, 0])
        assert math.isnan(float(false_negative_rate(y_pred, y_true)))

    def test_returns_backend_type(self, array_fn, array_type):
        """Results are instances of the input backend's type."""
        y_true = array_fn([0, 1])
        y_pred = array_fn([1, 0])
        assert isinstance(false_positive_rate(y_pred, y_true), array_type)
        assert isinstance(false_negative_rate(y_pred, y_true), array_type)

    def test_rejects_mismatched_lengths(self, array_fn):
        """y_true and y_pred must have the same number of samples."""
        y_true = array_fn([0, 1, 0])
        y_pred = array_fn([0, 1])
        with pytest.raises(ValueError, match="batch size"):
            false_positive_rate(y_pred, y_true)
        with pytest.raises(ValueError, match="batch size"):
            false_negative_rate(y_pred, y_true)
