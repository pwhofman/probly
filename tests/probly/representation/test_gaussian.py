"""Tests for Numpy-based Gaussian distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.distribution.array_gaussian import ArrayGaussian


class MockGaussianArray(ArrayGaussian):
    """soon to be replaced."""

    def sample(self, size: int) -> np.ndarray:
        return np.zeros((size, *self.shape))


# Mock kann weg sobald sample in common / distribution da ist


def test_array_gaussian_initialization_valid() -> None:
    """Test standard initialization with valid numpy arrays."""
    mean = np.array([0.0, 1.0])
    var = np.array([1.0, 0.5])

    dist = MockGaussianArray(mean=mean, var=var)

    np.testing.assert_array_equal(dist.mean, mean)
    np.testing.assert_array_equal(dist.var, var)
    assert dist.type == "gaussian"
    assert dist.device == "cpu"


def test_array_gaussian_initialization() -> None:
    """Test that values are treated correctly in __init__ (type wise)."""
    mean = np.array([0, 1])
    var = np.array([1, 2])

    dist = MockGaussianArray(mean=mean, var=var)

    assert dist.mean.dtype == np.float64
    assert dist.var.dtype == np.float64


def test_array_gaussian_raises_on_shape_mismatch() -> None:
    """Test something."""
    mean = np.zeros((5,))
    var = np.ones((4,))

    with pytest.raises(ValueError, match="mean and var must have same shape"):
        MockGaussianArray(mean=mean, var=var)


@pytest.mark.parametrize("invalid_var", [0.0, -0.1, -5.0])
def test_array_gaussian_raises_on_non_positive_variance(invalid_var: float) -> None:
    """Test something."""
    mean = np.array([0.0, 0.0])
    var = np.array([1.0, invalid_var])

    with pytest.raises(ValueError, match="Variance must be positive"):
        MockGaussianArray(mean=mean, var=var)


def test_from_parameters_creates_instance() -> None:
    """Test the from_parameters factory method."""
    mean_list = [1.0, 2.0]
    var_list = [0.5, 0.5]

    dist = MockGaussianArray.from_parameters(mean=mean_list, var=var_list)

    assert isinstance(dist, MockGaussianArray)

    np.testing.assert_array_equal(dist.mean, np.array(mean_list))


def test_from_parameters_basic() -> None:
    """Test something."""
    mean = [0.0, 1.0, 2.0]
    var = [1.0, 4.0, 9.0]

    g = MockGaussianArray.from_parameters(mean, var)

    assert isinstance(g, MockGaussianArray)

    np.testing.assert_array_equal(g.mean, np.array(mean, dtype=float))
    np.testing.assert_array_equal(g.var, np.array(var, dtype=float))

    assert g.mean.dtype == float
    assert g.var.dtype == g.mean.dtype


def test_array_properties() -> None:
    """Test shape, ndim, size delegation."""
    shape = (2, 3)
    mean = np.zeros(shape)
    var = np.ones(shape)

    dist = MockGaussianArray(mean, var)

    assert dist.shape == shape
    assert dist.ndim == 2
    assert dist.size == 6
    assert dist.__array_namespace__ is np


def test_transpose_property() -> None:
    """Test the .T property."""
    mean = np.array([[1.0, 2.0], [3.0, 4.0]])
    var = np.array([[0.1, 0.2], [0.3, 0.4]])

    dist = MockGaussianArray(mean, var)
    transposed = dist.T

    # wsl keine mock types mehr nötig
    # Das Transponierte Objekt ist wieder vom Typ der aufrufenden Klasse (Mock Objekt)
    assert isinstance(transposed, MockGaussianArray)
    assert transposed.shape == (2, 2)
    np.testing.assert_array_equal(transposed.mean, mean.T)


def test_matrix_transpose_property() -> None:
    """Test the .mT property."""
    shape = (2, 3, 4)
    rng = np.random.default_rng(seed=42)
    mean = rng.random(shape)
    var = np.ones(shape)

    dist = MockGaussianArray(mean, var)
    mt_dist = dist.mT

    assert mt_dist.shape == (2, 4, 3)
    # Check logic matches numpy's matrix_transpose
    expected_mean = np.matrix_transpose(mean)
    np.testing.assert_array_equal(mt_dist.mean, expected_mean)
