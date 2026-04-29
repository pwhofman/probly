"""Tests for Numpy-based Gaussian distribution representation."""

from __future__ import annotations

import pytest
import torch

from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution
from probly.representation.sample.torch import TorchSample


def test_torch_gaussian_initialization_valid() -> None:
    """Test standard initialization with valid numpy arrays aswell as types."""
    mean = torch.tensor([0.0, 1.0])
    var = torch.tensor([1.0, 0.5])

    dist = TorchGaussianDistribution(mean=mean, var=var)

    torch.testing.assert_close(dist.mean, mean)
    torch.testing.assert_close(dist.var, var)
    assert dist.type == "gaussian"
    assert dist.mean.dtype == torch.float32
    assert dist.var.dtype == torch.float32


def test_torch_gaussian_raises_on_shape_mismatch() -> None:
    """Test if the function does raise a ValueError upon wrong initialization."""
    mean = torch.zeros((5,))
    var = torch.ones((4,))

    with pytest.raises(ValueError, match="mean and var must have same shape"):
        TorchGaussianDistribution(mean=mean, var=var)


@pytest.mark.parametrize("invalid_var", [0.0, -0.1, -5.0])
def test_torch_gaussian_raises_on_non_positive_variance(invalid_var: float) -> None:
    """Test if the function does raise a ValueError upon using a negative variance."""
    mean = torch.tensor([0.0, 0.0])
    var = torch.tensor([1.0, invalid_var])

    with pytest.raises(ValueError, match="Variance must be positive"):
        TorchGaussianDistribution(mean=mean, var=var)


def test_from_parameters_creates_instance() -> None:
    """Test the from_parameters factory method."""
    mean_list = torch.tensor([1.0, 2.0])
    var_list = torch.tensor([0.5, 0.5])

    dist = TorchGaussianDistribution(mean=mean_list, var=var_list)

    assert isinstance(dist, TorchGaussianDistribution)

    torch.testing.assert_close(dist.mean, torch.tensor(mean_list, dtype=torch.float32))
    torch.testing.assert_close(dist.var, torch.tensor(var_list, dtype=torch.float32))


def test_torch_properties() -> None:
    """Test shape, ndim, size delegation."""
    shape = (2, 3)
    mean = torch.zeros(shape)
    var = torch.ones(shape)

    dist = TorchGaussianDistribution(mean, var)

    assert dist.shape == shape
    assert dist.ndim == 2


def test_transpose_property() -> None:
    """Test the .T property."""
    mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    var = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    dist = TorchGaussianDistribution(mean, var)
    transposed = dist.T

    assert isinstance(transposed, TorchGaussianDistribution)
    assert transposed.shape == (2, 2)
    torch.testing.assert_close(transposed.mean, mean.T)


def test_matrix_transpose_property() -> None:
    """Test the .mT property."""
    shape = (2, 3, 4)
    mean = torch.zeros(shape)
    var = torch.ones(shape)
    dist = TorchGaussianDistribution(mean, var)

    t_dist = dist.T

    assert t_dist.shape == (4, 3, 2)
    expected_mean = mean.T
    torch.testing.assert_close(t_dist.mean, expected_mean)


def test_sample_function() -> None:
    """Test the sampling function returns."""
    shape = (2,)
    dist = TorchGaussianDistribution(torch.zeros(shape), torch.ones(shape))

    n_samples = 4
    samples = dist.sample(n_samples)

    assert isinstance(samples, TorchSample)
    assert samples.tensor.shape == (n_samples, *shape)
    assert samples.sample_axis == 0


def test_sample_statistics() -> None:
    """Check if the samples actually follow the Gaussian distribution statistically."""
    mean_val = 10.0
    var_val = 4.0
    dist = TorchGaussianDistribution(torch.tensor([mean_val]), torch.tensor([var_val]))

    n_samples = 100000
    sample_wrapper = dist.sample(n_samples)
    samples = sample_wrapper.tensor

    assert torch.mean(samples) == pytest.approx(mean_val, abs=0.1)
    assert torch.var(samples) == pytest.approx(var_val, abs=0.1)


def test_slice() -> None:
    """Test slicing via __getitem__ returns a new TorchGaussian."""
    mean = torch.tensor([10.0, 20.0, 30.0])
    var = torch.tensor([1.0, 1.0, 1.0])
    dist = TorchGaussianDistribution(mean, var)

    sliced = dist[:2]

    assert isinstance(sliced, TorchGaussianDistribution)
    assert sliced.shape == (2,)
    torch.testing.assert_close(sliced.mean, torch.tensor([10.0, 20.0]))
    torch.testing.assert_close(sliced.var, torch.tensor([1.0, 1.0]))
