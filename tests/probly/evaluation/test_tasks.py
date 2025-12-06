"""Tests for the tasks module."""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.tasks import (
    fpr_at_tpr,
    out_of_distribution_detection,
    out_of_distribution_detection_aupr,
    out_of_distribution_detection_fnr_at_95,
    selective_prediction,
)


def test_selective_prediction_shapes() -> None:
    rng = np.random.default_rng()
    auroc, bin_losses = selective_prediction(rng.random(10), rng.random(10), n_bins=5)
    assert isinstance(auroc, float)
    assert isinstance(bin_losses, np.ndarray)
    assert bin_losses.shape == (5,)


def test_selective_prediction_order() -> None:
    criterion = np.linspace(0, 1, 10)
    losses = np.linspace(0, 1, 10)
    _, bin_losses = selective_prediction(criterion, losses, n_bins=5)
    assert np.all(np.diff(bin_losses) <= 0)


def test_selective_prediction_too_many_bins() -> None:
    rng = np.random.default_rng()
    with pytest.raises(ValueError, match="The number of bins can not be larger than the number of elements criterion"):
        selective_prediction(rng.random(5), rng.random(5), n_bins=10)


def test_out_of_distribution_detection_shape() -> None:
    rng = np.random.default_rng()
    auroc = out_of_distribution_detection(rng.random(10), rng.random(10))
    assert isinstance(auroc, float)


def test_out_of_distribution_detection_order() -> None:
    in_distribution = np.linspace(0, 1, 10)
    out_distribution = np.linspace(0, 1, 10) + 1
    auroc = out_of_distribution_detection(in_distribution, out_distribution)
    assert np.isclose(auroc, 0.995)


def test_out_of_distribution_detection_aupr_shape() -> None:
    """Test that AUPR OOD detection returns a float."""
    rng = np.random.default_rng()
    aupr = out_of_distribution_detection_aupr(rng.random(10), rng.random(10))
    assert isinstance(aupr, float)


def test_out_of_distribution_detection_aupr_order() -> None:
    """Test that AUPR OOD detection gives high score when OOD clearly differs from ID."""
    in_distribution = np.linspace(0, 1, 10)
    out_distribution = np.linspace(0, 1, 10) + 1  # clearly separated distributions
    aupr = out_of_distribution_detection_aupr(in_distribution, out_distribution)
    assert aupr > 0.99


def test_fpr_at_tpr_simple_case() -> None:
    in_scores = np.array([0.1, 0.2, 0.6, 0.7])
    out_scores = np.array([0.3, 0.4, 0.8, 0.9])

    fpr = fpr_at_tpr(in_scores, out_scores, tpr_target=0.95)

    assert np.isclose(fpr, 0.5)


def test_fpr_at_tpr_invalid_tpr_target() -> None:
    in_scores = np.array([0.1, 0.2])
    out_scores = np.array([0.8, 0.9])

    msg = r"tpr_target must be in the interval \(0, 1]"

    with pytest.raises(ValueError, match=msg):
        fpr_at_tpr(in_scores, out_scores, tpr_target=0.0)

    with pytest.raises(ValueError, match=msg):
        fpr_at_tpr(in_scores, out_scores, tpr_target=1.1)


def test_fpr_at_tpr_perfect_separation() -> None:
    in_scores = np.array([0.1, 0.2, 0.3, 0.4])
    out_scores = np.array([0.8, 0.9, 1.0, 1.1])

    fpr = fpr_at_tpr(in_scores, out_scores, tpr_target=0.95)

    assert np.isclose(fpr, 0.0)


def test_fnr_at_95_returns_float() -> None:
    """Tests if the funtion returns floats."""
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([0.8, 0.9, 1.0])

    fnr = out_of_distribution_detection_fnr_at_95(in_distribution, out_distribution)

    assert isinstance(fnr, float)


def test_fnr_zero_when_perfect_separation() -> None:
    """If ID scores are clearly lower than OOD scores, FN should be 0."""
    in_distribution = np.array([0.1, 0.2, 0.3])
    out_distribution = np.array([1.0, 0.9, 0.8])

    fnr = out_of_distribution_detection_fnr_at_95(in_distribution, out_distribution)
    assert fnr == 0.0


def test_fnr_one_when_in_and_out_fully_reversed() -> None:
    """If OOD scores are completely lower than all ID scores.

    Then at 95% TPR all OOD samples are misclassified => FNR = 1.
    """
    in_distribution = np.array([1.0, 0.9, 0.8])
    out_distribution = np.array([0.1, 0.2, 0.3])

    fnr = out_of_distribution_detection_fnr_at_95(in_distribution, out_distribution)
    assert fnr == 1.0


def test_fnr_with_partial_overlap() -> None:
    """With overlapping distributions, the FNR should be between 0 and 1."""
    in_distribution = np.array([0.1, 0.4, 0.6])
    out_distribution = np.array([0.3, 0.5, 0.9])

    fnr = out_of_distribution_detection_fnr_at_95(in_distribution, out_distribution)
    assert 0.0 <= fnr <= 1.0


def test_single_element_arrays() -> None:
    """Edge case: one ID sample and one OOD sample."""
    in_distribution = np.array([0.2])
    out_distribution = np.array([0.9])

    fnr = out_of_distribution_detection_fnr_at_95(in_distribution, out_distribution)
    assert fnr == 0.0
