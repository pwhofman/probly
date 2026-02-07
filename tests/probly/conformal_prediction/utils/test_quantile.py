"""Tests for quantile calculation utilities."""

from __future__ import annotations

import numpy as np

from probly.conformal_prediction.utils.quantile import calculate_quantile, calculate_weighted_quantile


def test_calculate_quantile() -> None:
    """Test quantile calculation."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # test different alpha values
    assert np.isclose(calculate_quantile(scores, 0.1), 0.5)  # 90% coverage
    assert np.isclose(calculate_quantile(scores, 0.5), 0.3)  # 50% coverage (inverted cdf)
    assert np.isclose(calculate_quantile(scores, 0.9), 0.1)  # 10% coverage (inverted cdf)


def test_calculate_weighted_quantile() -> None:
    """Test weighted quantile calculation."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = calculate_weighted_quantile(values, 0.5, sample_weight=None)
    assert q == 3.0
