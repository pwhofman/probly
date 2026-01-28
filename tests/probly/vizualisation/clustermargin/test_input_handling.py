"""Tests for input_handling."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

from probly.visualization.clustermargin.clustervisualizer import _2_cluster_to_x, _2_cluster_to_y, _check_shape

mpl.use("Agg")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def create_cluster1(rng: np.random.Generator) -> np.ndarray:
    mu = [0, 0]
    sigma = [[0.05, 0], [0, 0.5]]

    return rng.multivariate_normal(mu, sigma, size=10)


@pytest.fixture
def create_cluster2(rng: np.random.Generator) -> np.ndarray:
    mu = [1, 1]
    sigma = [[0.05, 0], [0, 0.5]]

    return rng.multivariate_normal(mu, sigma, size=10)


def test_check_shape_valid_input() -> None:
    """Test that valid input is returned as is."""
    data = np.array([[0, 0], [1, 1]])
    result = _check_shape(data)
    np.testing.assert_array_equal(result, data)


def test_check_shape_raises_on_list() -> None:
    """Test TypeError if input is not a numpy array."""
    with pytest.raises(TypeError, match="Input must be a NumPy Array"):
        _check_shape([0.5, 0.5])


def test_check_shape_raises_on_empty() -> None:
    """Test ValueError if input is empty."""
    with pytest.raises(ValueError, match="Input must not be empty"):
        _check_shape(np.array([]))


def test_check_shape_raises_on_1d() -> None:
    """Test ValueError if input is 1D (must be shape (n_samples, 2)."""
    with pytest.raises(ValueError, match=r"Input must have shape \(n_samples, 2\)"):
        _check_shape(np.array([0.5, 0.5]))


def test_check_shape_second_dim_not_2() -> None:
    """Tests ValueError if second dimension is not 2."""
    with pytest.raises(ValueError, match=r"Input must have shape \(n_samples, 2\)"):
        _check_shape(np.array([[1, 0, 2], [0, 1, 2]]))


def test_2_cluster_to_y_creates_1d_array(create_cluster1: np.ndarray, create_cluster2: np.ndarray) -> None:
    """Test that input is converted to a 1D array."""
    data = _2_cluster_to_y(create_cluster1, create_cluster2)
    assert data.ndim == 1


def test_2_cluster_to_y_consists_of_0_and_1(create_cluster1: np.ndarray, create_cluster2: np.ndarray) -> None:
    """Tests that the created array only consists of 0s and 1."""
    data = _2_cluster_to_y(create_cluster1, create_cluster2)
    for value in data:
        assert value in (0, 1)


def test_2_cluster_to_x_creates_2d_array(create_cluster1: np.ndarray, create_cluster2: np.ndarray) -> None:
    """Test that input is converted to a 2D array with shape (n_samples, 2)."""
    data = _2_cluster_to_x(create_cluster1, create_cluster2)
    assert data.ndim == 2


def test_2_cluster_to_x_dim_2_equal_2(create_cluster1: np.ndarray, create_cluster2: np.ndarray) -> None:
    """Test that input is converted to a 2D array with shape (n_samples, 2)."""
    data = _2_cluster_to_x(create_cluster1, create_cluster2)
    assert data.shape[1] == 2
