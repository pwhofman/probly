"""Tests for the tasks module."""

from __future__ import annotations

import numpy as np
import pytest

from probly.tasks import out_of_distribution_detection, selective_prediction


def test_selective_prediction_shapes():
    rng = np.random.default_rng()
    auroc, bin_losses = selective_prediction(rng.random(10), rng.random(10), n_bins=5)
    assert isinstance(auroc, float)
    assert isinstance(bin_losses, np.ndarray)
    assert bin_losses.shape == (5,)


def test_selective_prediction_order():
    criterion = np.linspace(0, 1, 10)
    losses = np.linspace(0, 1, 10)
    _, bin_losses = selective_prediction(criterion, losses, n_bins=5)
    assert np.all(np.diff(bin_losses) <= 0)


def test_selective_prediction_too_many_bins():
    rng = np.random.default_rng()
    with pytest.raises(ValueError, match="The number of bins can not be larger than the number of elements criterion"):
        selective_prediction(rng.random(5), rng.random(5), n_bins=10)


def test_out_of_distribution_detection_shape():
    rng = np.random.default_rng()
    auroc = out_of_distribution_detection(rng.random(10), rng.random(10))
    assert isinstance(auroc, float)


def test_out_of_distribution_detection_order():
    in_distribution = np.linspace(0, 1, 10)
    out_distribution = np.linspace(0, 1, 10) + 1
    auroc = out_of_distribution_detection(in_distribution, out_distribution)
    assert auroc == 0.995
