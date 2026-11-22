from __future__ import annotations

from common.evaluate import rejection_curve
import numpy as np


def test_perfect_scores_drop_to_zero_error():
    correct = [True] * 6 + [False] * 4
    uncertainty = [0.0] * 6 + [1.0] * 4
    rejected, error = rejection_curve(uncertainty, correct)
    assert rejected[0] == 0
    assert np.isclose(error[0], 0.4)
    assert np.isclose(error[4], 0.0)


def test_curve_has_one_value_per_bin():
    rejected, error = rejection_curve(np.arange(20.0), [True, False] * 10, n_bins=5)
    assert len(rejected) == len(error) == 5
