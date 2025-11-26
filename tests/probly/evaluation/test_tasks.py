"""Tests for the tasks module."""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.tasks import out_of_distribution_detection_fpr_at_95_tpr, selective_prediction, out_of_distribution_detection_fnr_at_95


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
    assert auroc == 0.995



def test_fpr_at_95_tpr_returns_float():
    """Tests if the function returns floats."""
    in_dist = np.zeros(20)
    out_dist = np.ones(20)
    """No random floats to reduce the possibility of the code crashing."""

    fpr = out_of_distribution_detection_fpr_at_95_tpr(in_dist, out_dist)

    assert isinstance(fpr, float)


def test_fpr_at_95_tpr_raises_if_no_exact_95_point():
    """Test that function raises an error if TPR=0.95 does not exist."""
    in_dist = np.array([0.1, 0.2, 0.3])
    out_dist = np.array([0.4, 0.5, 0.6])

    with pytest.raises(IndexError):
        out_of_distribution_detection_fpr_at_95_tpr(in_dist, out_dist)


def test_fpr_at_95_tpr_perfect_separation():
    """Test if FTP@95TPR OOD values are greater than ID values."""
    in_dist = np.array([0.1, 0.2, 0.3, 0.4])
    out_dist = np.array([0.9, 0.95, 0.96, 0.97])

    result = out_of_distribution_detection_fpr_at_95_tpr(in_dist, out_dist)

    assert result == 1.0


def test_fpr_at_95_tpr_complete_overlap():
    """Tests if FTP@95TPR OOD- and ID-values are identical."""
    in_dist = np.array([0.5, 0.5, 0.5, 0.5])
    out_dist = np.array([0.5, 0.5, 0.5, 0.5])

    result = out_of_distribution_detection_fpr_at_95_tpr(in_dist, out_dist)

    assert result == 0.0


def test_fnr_at_95_returns_float():
    """Tests if the funtion returns floats."""
    in_scores = np.array([0.1, 0.2, 0.3])
    out_scores = np.array([0.8, 0.9, 1.0])

    fnr = out_of_distribution_detection_fnr_at_95(in_scores, out_scores)

    assert isinstance(fnr, float)

def test_fnr_zero_when_perfect_separation():
    """If ID scores are clearly lower than OOD scores, FN should be 0."""
    in_scores = np.array([0.1, 0.2, 0.3])
    out_scores = np.array([1.0, 0.9, 0.8])

    fnr = out_of_distribution_detection_fnr_at_95(in_scores, out_scores)
    assert fnr == 0.0


def test_fnr_one_when_in_and_out_fully_reversed():
    """If OOD scores are completely lower than all ID scores, 
    then at 95% TPR all OOD samples are misclassified => FNR = 1."""
    in_scores = np.array([0.9, 0.8, 1.0])
    out_scores = np.array([0.1, 0.2, 0.3])

    fnr = out_of_distribution_detection_fnr_at_95(in_scores, out_scores)
    assert fnr == 1.0


def test_fnr_with_partial_overlap():
    """With overlapping distributions, the FNR should be between 0 and 1."""
    in_scores = np.array([0.1, 0.4, 0.6])
    out_scores = np.array([0.3, 0.5, 0.9])

    fnr = out_of_distribution_detection_fnr_at_95(in_scores, out_scores)
    assert 0.0 <= fnr <= 1.0


def test_single_element_arrays():
    """Edge case: one ID sample and one OOD sample."""
    in_scores = np.array([0.2])
    out_scores = np.array([0.9])

    fnr = out_of_distribution_detection_fnr_at_95(in_scores, out_scores)
    assert fnr == 0.0  # OOD is correctly classified as OOD
