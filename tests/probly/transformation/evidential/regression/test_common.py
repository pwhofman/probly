"""Tests for the general evidential regression transformation API."""

from __future__ import annotations

from probly.transformation.evidential.regression import evidential_regression


def test_evidential_regression_exists() -> None:
    """Test that the evidential_regression function can be imported."""
    assert callable(evidential_regression)
