"""Tests for input_handling."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes
import numpy as np
import pytest

mpl.use("Agg")

from probly.visualization.credal.input_handling import (
    check_num_classes,
    check_shape,
    dispatch_plot,
    normalize_input,
)


def test_check_shape_valid_input() -> None:
    """Test that valid input is returned as is."""
    data = np.array([[0.2, 0.8], [0.5, 0.5]])
    result = check_shape(data)
    np.testing.assert_array_equal(result, data)


def test_check_shape_raises_on_list() -> None:
    """Test TypeError if input is not a numpy array."""
    with pytest.raises(TypeError, match="Input must be a NumPy Array"):
        check_shape([0.5, 0.5])


def test_check_shape_raises_on_empty() -> None:
    """Test ValueError if input is empty."""
    with pytest.raises(ValueError, match="Input must not be empty"):
        check_shape(np.array([]))


def test_check_shape_raises_on_1d() -> None:
    """Test ValueError if input is 1D (must be at least 2D)."""
    with pytest.raises(ValueError, match="Input must be at least 2D"):
        check_shape(np.array([0.5, 0.5]))


def test_check_shape_raises_sum_not_one() -> None:
    """Test ValueError if probabilities do not sum to 1."""
    with pytest.raises(ValueError, match="must sum to 1"):
        check_shape(np.array([[0.1, 0.1]]))


def test_check_shape_raises_negative() -> None:
    """Test ValueError if probabilities are negative."""
    with pytest.raises(ValueError, match="must be positive"):
        check_shape(np.array([[1.2, -0.2]]))


def test_check_shape_raises_single_class() -> None:
    """Test ValueError if there is only one class."""
    with pytest.raises(ValueError, match="more than one class"):
        check_shape(np.array([[1.0]]))


def test_check_num_classes() -> None:
    """Test class counting logic."""
    data = np.zeros((5, 3))
    assert check_num_classes(data) == 3


def test_normalize_input_3d_to_2d() -> None:
    """Test that 3D input is flattened to 2D."""
    data = np.zeros((2, 2, 3))
    result = normalize_input(data)
    assert result.ndim == 2
    assert result.shape == (4, 3)


def test_dispatch_plot_2_classes_integration() -> None:
    """Test execution for 2 classes (IntervalVisualizer).Checks if a matplotlib Axes object is returned."""
    data = np.array([[0.3, 0.7], [0.4, 0.6]])
    ax = dispatch_plot(data, choice="MLE")

    assert isinstance(ax, Axes)
    assert ax.get_title() == "Credal Plot (2 Classes)"


def test_dispatch_plot_3_classes_integration() -> None:
    """Test execution for 3 classes (TernaryVisualizer). Checks if a matplotlib Axes object is returned."""
    data = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    ax = dispatch_plot(data, choice="Credal")
    assert ax is not None


def test_dispatch_plot_multi_classes_integration() -> None:
    """Test execution for >3 classes (MultiVisualizer)."""
    data = np.array([[0.25, 0.25, 0.25, 0.25]])
    labels = ["A", "B", "C", "D"]

    ax = dispatch_plot(data, labels=labels, choice="Probability")

    assert ax is not None
