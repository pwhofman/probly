"""Test for input handling."""

from __future__ import annotations

import numpy as np
import pytest

from probly.visualization.input_handling import *  # noqa: F403


def test_check_num_classes_2d_array() -> None:
    data = np.zeros((5, 3))
    n_classes = check_num_classes(data)
    assert n_classes == 3

def test_check_num_classes_3d_array() -> None:
    data = np.zeros((2, 4, 7))
    n_classes = check_num_classes(data)
    assert n_classes == 7

def test_normalize_input_keeps_2d_shape() -> None:
    data = np.ones((4, 3)) / 3
    result = normalize_input(data)
    assert result.shape == (4, 3)
    assert result is data

def test_normalize_input_flattens_3d_to_2d() -> None:
    data = np.ones((2, 4, 3)) / 3
    result = normalize_input(data)
    assert result.shape == (8, 3)

def test_dispatch_plot_raises_for_invalid_probabilities() -> None:
    """If value is not 1.0, Error will be raised."""
    data = np.array([[0.5, 0.4]])
    with pytest.raises(ValueError):
        dispatch_plot(data)
