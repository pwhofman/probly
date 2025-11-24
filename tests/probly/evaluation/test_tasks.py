"""Tests for the tasks module."""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.tasks import out_of_distribution_detection, selective_prediction, fpr_at_tpr


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


def test_fpr_at_tpr_simple_case() -> None:
    # Scores: higher = more OOD-like (positive class)
    in_scores = np.array([0.1, 0.2, 0.6, 0.7])
    out_scores = np.array([0.3, 0.4, 0.8, 0.9])

    # Manually checked via sklearn.roc_curve:
    # FPR: [0. , 0. , 0. , 0.5, 0.5, 1. ]
    # TPR: [0. , 0.25, 0.5, 0.5, 1. , 1. ]
    # For target TPR=0.95, indices with TPR >= 0.95 are [4, 5]
    # We take the first index → FPR = 0.5
    fpr = fpr_at_tpr(in_scores, out_scores, tpr_target=0.95)

    assert np.isclose(fpr, 0.5)
    
def test_fpr_at_tpr_invalid_tpr_target() -> None:
    in_scores = np.array([0.1, 0.2])
    out_scores = np.array([0.8, 0.9])

    with pytest.raises(ValueError):
        fpr_at_tpr(in_scores, out_scores, tpr_target=0.0)

    with pytest.raises(ValueError):
        fpr_at_tpr(in_scores, out_scores, tpr_target=1.1)
        
        
def test_fpr_at_tpr_perfect_separation() -> None:
    # Perfect separation: all OOD scores > all ID scores
    in_scores = np.array([0.1, 0.2, 0.3, 0.4])
    out_scores = np.array([0.8, 0.9, 1.0, 1.1])

    fpr = fpr_at_tpr(in_scores, out_scores, tpr_target=0.95)

    # We can reach TPR=1.0 while still having FPR=0.0
    assert np.isclose(fpr, 0.0)