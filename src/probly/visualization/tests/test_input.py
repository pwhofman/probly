"""Test for input handling."""

from __future__ import annotations

import numpy as np
import pytest

from probly.visualization.input_handling import check_num_classes, dispatch_plot, normalize_input


def test_check_num_classes_2d_array() -> None:
    """Testing if classes are staying the same for 2d."""
    data = np.zeros((5, 3))
    n_classes = check_num_classes(data)
    assert n_classes == 3  # noqa: S101


def test_check_num_classes_3d_array() -> None:
    """Testing if classes are staying the same for 3d."""
    data = np.zeros((2, 4, 7))
    n_classes = check_num_classes(data)
    assert n_classes == 7  # noqa: S101


def test_normalize_input_keeps_2d_shape() -> None:
    """Testing if normalizing input keeps shape."""
    data = np.ones((4, 3)) / 3
    result = normalize_input(data)
    assert result.shape == (4, 3)  # noqa: S101
    assert result is data  # noqa: S101


def test_normalize_input_flattens_3d_to_2d() -> None:
    """Testing if normalising flattens correctly."""
    data = np.ones((2, 4, 3)) / 3
    result = normalize_input(data)
    assert result.shape == (8, 3)  # noqa: S101


def test_dispatch_plot_raises_for_invalid_probabilities() -> None:
    """If value is not 1.0, Error will be raised."""
    data = np.array([[0.5, 0.4]])
    with pytest.raises(ValueError, match=r"The probabilities of each class must sum to 1"):
        dispatch_plot(data)
