"""Simple test for APS common functions."""

from __future__ import annotations

import numpy as np

from probly.conformal_prediction.aps.common import calculate_nonconformity_score, calculate_quantile


def test_calculate_nonconformity_score_basic() -> None:
    """Basic test with simple data."""
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
        ],
    )
    labels = np.array([0, 1, 2])

    scores = calculate_nonconformity_score(probs, labels)
    expected = np.array([0.5, 0.7, 0.6])

    assert np.allclose(scores, expected)


def test_calculate_nonconformity_score_edge_cases() -> None:
    """Test edge cases."""
    probs = np.array([[0.1, 0.9]])
    labels = np.array([1])
    scores = calculate_nonconformity_score(probs, labels)
    assert np.allclose(scores, [0.9])

    # uniform distribution
    probs = np.array([[0.33, 0.33, 0.34]])
    labels = np.array([2])
    scores = calculate_nonconformity_score(probs, labels)
    assert np.allclose(scores, [0.34])


def test_calculate_quantile() -> None:
    """Test quantile calculation."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # test dif alpha values
    assert np.isclose(calculate_quantile(scores, 0.1), 0.5)  # 90% coverage
    assert np.isclose(calculate_quantile(scores, 0.5), 0.3)  # 50% coverage
    assert np.isclose(calculate_quantile(scores, 0.9), 0.1)  # 10% coverage


def test_calculate_nonconformity_score_ties() -> None:
    """Test with tied probabilities."""
    probs = np.array(
        [
            [0.4, 0.4, 0.2],
            [0.3, 0.3, 0.4],  # Classes 0 and 1 have equal probability
        ],
    )
    labels = np.array([0, 2])

    scores = calculate_nonconformity_score(probs, labels)

    # With ties: score = cumulative probability up to the first position
    # where the true class appears in sorted order

    # Expected:
    # Sample 0: true class 0 appears at position 0 (ties sorted arbitrarily)
    #           cumulative prob at position 0 = 0.4
    # Sample 1: true class 2 appears at position 0 (highest probability)
    #           cumulative prob at position 0 = 0.4

    expected = np.array([0.4, 0.4])
    assert np.allclose(scores, expected), f"Expected {expected}, got {scores}"

    # Additional assertion to verify scores are within valid range
    assert np.all(scores >= 0), "Scores should be non-negative"
    assert np.all(scores <= 1), "Scores should not exceed 1"


if __name__ == "__main__":
    test_calculate_nonconformity_score_basic()
    test_calculate_quantile()
    test_calculate_nonconformity_score_edge_cases()
    test_calculate_nonconformity_score_ties()
