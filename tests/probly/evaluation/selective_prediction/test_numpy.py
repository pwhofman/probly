"""NumPy backend tests for selective prediction."""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.selective_prediction import selective_prediction


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


def test_selective_prediction_exact_values() -> None:
    criterion = np.array([3.0, 1.0, 2.0, 0.0])
    losses = np.array([30.0, 10.0, 20.0, 0.0])
    aurc, bin_losses = selective_prediction(criterion, losses, n_bins=2)
    np.testing.assert_array_equal(bin_losses, [15.0, 5.0])
    assert aurc == 10.0


def test_selective_prediction_too_many_bins() -> None:
    rng = np.random.default_rng()
    with pytest.raises(ValueError, match="The number of bins can not be larger than the number of elements criterion"):
        selective_prediction(rng.random(5), rng.random(5), n_bins=10)
