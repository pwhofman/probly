"""Tests for Numpy-based Gaussian distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.distribution.array_gaussian import ArrayGaussian
from probly.representation.sampling import ArraySample


def test_array_gaussian_initialization_valid() -> None:
    """Test standard initialization with valid numpy arrays aswell as types."""
    mean = np.array([0.0, 1.0])
    var = np.array([1.0, 0.5])

    dist = ArrayGaussian(mean=mean, var=var)

    np.testing.assert_array_equal(dist.mean, mean)
    np.testing.assert_array_equal(dist.var, var)
    assert dist.type == "gaussian"
    assert dist.device == "cpu"
    assert dist.mean.dtype == np.float64
    assert dist.var.dtype == np.float64


def test_array_gaussian_raises_on_shape_mismatch() -> None:
    """Test if the function does raise a ValueError upon wrong initialization."""
    mean = np.zeros((5,))
    var = np.ones((4,))

    with pytest.raises(ValueError, match="mean and var must have same shape"):
        ArrayGaussian(mean=mean, var=var)


@pytest.mark.parametrize("invalid_var", [0.0, -0.1, -5.0])
def test_array_gaussian_raises_on_non_positive_variance(invalid_var: float) -> None:
    """Test if the function does raise a ValueError upon using a negative variance."""
    mean = np.array([0.0, 0.0])
    var = np.array([1.0, invalid_var])

    with pytest.raises(ValueError, match="Variance must be positive"):
        ArrayGaussian(mean=mean, var=var)


def test_from_parameters_creates_instance() -> None:
    """Test the from_parameters factory method."""
    mean_list = [1.0, 2.0]
    var_list = [0.5, 0.5]

    dist = ArrayGaussian.from_parameters(mean=mean_list, var=var_list)

    assert isinstance(dist, ArrayGaussian)

    np.testing.assert_array_equal(dist.mean, np.array(mean_list, dtype=float))
    np.testing.assert_array_equal(dist.var, np.array(var_list, dtype=float))



def test_array_properties() -> None:
    """Test shape, ndim, size delegation."""
    shape = (2, 3)
    mean = np.zeros(shape)
    var = np.ones(shape)

    dist = ArrayGaussian(mean, var)

    assert dist.shape == shape
    assert dist.ndim == 2
    assert dist.size == 6
    assert dist.__array_namespace__() is np


def test_transpose_property() -> None:
    """Test the .T property."""
    mean = np.array([[1.0, 2.0], [3.0, 4.0]])
    var = np.array([[0.1, 0.2], [0.3, 0.4]])

    dist = ArrayGaussian(mean, var)
    transposed = dist.T


    assert isinstance(transposed, ArrayGaussian)
    assert transposed.shape == (2, 2)
    np.testing.assert_array_equal(transposed.mean, mean.T)


def test_matrix_transpose_property() -> None:
    """Test the .mT property."""
    shape = (2, 3, 4)
    mean = np.zeros(shape)
    var = np.ones(shape)
    dist = ArrayGaussian(mean, var)

    t_dist = dist.T

    assert t_dist.shape == (4, 3, 2)
    expected_mean = np.transpose(mean)
    np.testing.assert_array_equal(t_dist.mean, expected_mean)

def test_sample_function() -> None :
    """Test the sampling function returns."""
    shape = (2,)
    dist = ArrayGaussian(np.zeros(shape), np.ones(shape))

    n_samples = 4
    samples = dist.sample(n_samples)

    assert isinstance(samples, ArraySample)
    assert samples.array.shape == (n_samples, *shape)
    assert samples.sample_axis == 0

def test_sample_statistics() -> None:
    """
    Check if the samples actually follow the Gaussian distribution statistically.
    """
    mean_val = 10.0
    var_val = 4.0
    dist = ArrayGaussian(np.array([mean_val]), np.array([var_val]))

    n_samples = 100000
    sample_wrapper = dist.sample(n_samples)
    samples = sample_wrapper.array

    assert np.mean(samples) == pytest.approx(mean_val, abs=0.05)
    assert np.var(samples) == pytest.approx(var_val, abs=0.05)

def test_entropy() -> None:
    """Test if entropy calculation works properly."""
    mean = np.array([0])
    var = np.array([1])

    dist = ArrayGaussian(mean=mean, var=var)

    expected = 0.5 * np.log(2 * np.pi * np.e * var)
    assert dist.entropy == pytest.approx(expected)

def test_slice() -> None:
    """Test slicing via __getitem__ returns a new ArrayGaussian."""
    mean = np.array([10.0, 20.0, 30.0])
    var = np.array([1.0, 1.0, 1.0])
    dist = ArrayGaussian(mean, var)

    sliced = dist[:2]

    assert isinstance(sliced, ArrayGaussian)
    assert sliced.shape == (2,)
    np.testing.assert_array_equal(sliced.mean, [10.0, 20.0])
    np.testing.assert_array_equal(sliced.var, [1.0, 1.0])


def test_copy_method() -> None:
    """Test copying."""
    mean = np.array([1.0])
    var = np.array([1.0])
    dist = ArrayGaussian(mean, var)

    copied = dist.copy()

    assert copied == dist
    assert copied is not dist
    assert copied.mean is not dist.mean
