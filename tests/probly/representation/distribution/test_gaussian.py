"""Tests for Numpy-based Gaussian distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution
from probly.representation.sample import ArraySample


def test_array_gaussian_initialization_valid() -> None:
    """Test standard initialization with valid numpy arrays aswell as types."""
    mean = np.array([0.0, 1.0])
    var = np.array([1.0, 0.5])

    dist = ArrayGaussianDistribution(mean=mean, var=var)

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
        ArrayGaussianDistribution(mean=mean, var=var)


@pytest.mark.parametrize("invalid_var", [0.0, -0.1, -5.0])
def test_array_gaussian_raises_on_non_positive_variance(invalid_var: float) -> None:
    """Test if the function does raise a ValueError upon using a negative variance."""
    mean = np.array([0.0, 0.0])
    var = np.array([1.0, invalid_var])

    with pytest.raises(ValueError, match="Variance must be positive"):
        ArrayGaussianDistribution(mean=mean, var=var)


def test_from_parameters_creates_instance() -> None:
    """Test the from_parameters factory method."""
    mean_list = [1.0, 2.0]
    var_list = [0.5, 0.5]

    dist = ArrayGaussianDistribution(mean=mean_list, var=var_list)

    assert isinstance(dist, ArrayGaussianDistribution)

    np.testing.assert_array_equal(dist.mean, np.array(mean_list, dtype=float))
    np.testing.assert_array_equal(dist.var, np.array(var_list, dtype=float))


def test_array_properties() -> None:
    """Test shape, ndim, size delegation."""
    shape = (2, 3)
    mean = np.zeros(shape)
    var = np.ones(shape)

    dist = ArrayGaussianDistribution(mean, var)

    assert dist.shape == shape
    assert dist.ndim == 2
    assert dist.size == 6
    assert dist.__array_namespace__() is np


def test_std() -> None:
    """Test standard deviation calculation."""
    dist = ArrayGaussianDistribution(np.array([0.0, 1.0]), np.array([1.0, 4.0]))

    np.testing.assert_array_equal(dist.std, np.array([1.0, 2.0]))


def test_quantile() -> None:
    """Test Gaussian quantile calculation."""
    dist = ArrayGaussianDistribution(np.array([0.0, 1.0]), np.array([1.0, 4.0]))

    scalar_quantile = dist.quantile(0.5)
    vector_quantile = dist.quantile(np.array([0.5, 0.8413447]))

    np.testing.assert_allclose(scalar_quantile, dist.mean)
    assert vector_quantile.shape == (2, 2)
    np.testing.assert_allclose(vector_quantile[:, 0], dist.mean)


def test_transpose_property() -> None:
    """Test the .T property."""
    mean = np.array([[1.0, 2.0], [3.0, 4.0]])
    var = np.array([[0.1, 0.2], [0.3, 0.4]])

    dist = ArrayGaussianDistribution(mean, var)
    transposed = dist.T

    assert isinstance(transposed, ArrayGaussianDistribution)
    assert transposed.shape == (2, 2)
    np.testing.assert_array_equal(transposed.mean, mean.T)


def test_matrix_transpose_property() -> None:
    """Test the .mT property."""
    shape = (2, 3, 4)
    mean = np.zeros(shape)
    var = np.ones(shape)
    dist = ArrayGaussianDistribution(mean, var)

    t_dist = dist.T

    assert t_dist.shape == (4, 3, 2)
    expected_mean = np.transpose(mean)
    np.testing.assert_array_equal(t_dist.mean, expected_mean)


def test_sample_function() -> None:
    """Test the sampling function returns."""
    shape = (2,)
    dist = ArrayGaussianDistribution(np.zeros(shape), np.ones(shape))

    n_samples = 4
    samples = dist.sample(n_samples)

    assert isinstance(samples, ArraySample)
    assert samples.array.shape == (n_samples, *shape)
    assert samples.sample_axis == 0


def test_sample_statistics() -> None:
    """Check if the samples actually follow the Gaussian distribution statistically."""
    mean_val = 10.0
    var_val = 4.0
    dist = ArrayGaussianDistribution(np.array([mean_val]), np.array([var_val]))

    n_samples = 100000
    sample_wrapper = dist.sample(n_samples)
    samples = sample_wrapper.array

    assert np.mean(samples) == pytest.approx(mean_val, abs=0.1)
    assert np.var(samples) == pytest.approx(var_val, abs=0.1)


def test_entropy() -> None:
    """Test if entropy calculation works properly."""
    mean = np.array([0])
    var = np.array([1])

    dist = ArrayGaussianDistribution(mean=mean, var=var)

    expected = 0.5 * np.log(2 * np.pi * np.e * var)
    assert dist.entropy() == pytest.approx(expected)


def test_slice() -> None:
    """Test slicing via __getitem__ returns a new ArrayGaussian."""
    mean = np.array([10.0, 20.0, 30.0])
    var = np.array([1.0, 1.0, 1.0])
    dist = ArrayGaussianDistribution(mean, var)

    sliced = dist[:2]

    assert isinstance(sliced, ArrayGaussianDistribution)
    assert sliced.shape == (2,)
    np.testing.assert_array_equal(sliced.mean, [10.0, 20.0])
    np.testing.assert_array_equal(sliced.var, [1.0, 1.0])


def test_copy_method() -> None:
    """Test copying."""
    mean = np.array([1.0])
    var = np.array([1.0])
    dist = ArrayGaussianDistribution(mean, var)

    copied = dist.copy()

    assert copied == dist
    assert copied is not dist
    assert copied.mean is not dist.mean


class TestArrayGaussianDistribution:
    """Numpy-based Gaussian distribution."""

    def test_mismatched_shapes_raise(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        with pytest.raises(ValueError, match="same shape"):
            ArrayGaussianDistribution(mean=np.zeros((3,)), var=np.ones((4,)))

    def test_non_positive_var_raises(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        with pytest.raises(ValueError, match="must be positive"):
            ArrayGaussianDistribution(mean=np.zeros((3,)), var=np.zeros((3,)))

    def test_std_property(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.array([0.0, 1.0]), var=np.array([4.0, 9.0]))
        np.testing.assert_allclose(g.std, [2.0, 3.0])

    def test_quantile_scalar_q(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.array([0.0]), var=np.array([1.0]))
        # At q=0.5, the median equals the mean.
        np.testing.assert_allclose(g.quantile(0.5), [0.0], atol=1e-6)

    def test_quantile_array_q(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.array([0.0]), var=np.array([1.0]))
        result = g.quantile(np.array([0.5, 0.5]))
        # Two queries -> two outputs each of shape (1,).
        assert result.shape == (1, 2)

    def test_sample_returns_correct_shape(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.zeros((3,)), var=np.ones((3,)))
        samples = g.sample(num_samples=5)
        assert samples.array.shape == (5, 3)
        assert samples.sample_axis == 0

    def test_sample_uses_rng(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.zeros((2,)), var=np.ones((2,)))
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = g.sample(num_samples=3, rng=rng1)
        s2 = g.sample(num_samples=3, rng=rng2)
        # Same seed -> same samples.
        np.testing.assert_allclose(s1.array, s2.array)

    def test_array_dunder_stacks_mean_var(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.array([1.0, 2.0]), var=np.array([0.5, 0.5]))
        arr = np.asarray(g)
        # Last axis stacks mean and var.
        np.testing.assert_allclose(arr[..., 0], [1.0, 2.0])
        np.testing.assert_allclose(arr[..., 1], [0.5, 0.5])

    def test_addition_of_two_gaussians(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g1 = ArrayGaussianDistribution(mean=np.array([1.0]), var=np.array([2.0]))
        g2 = ArrayGaussianDistribution(mean=np.array([3.0]), var=np.array([5.0]))
        result = g1 + g2
        # Means add, variances add.
        np.testing.assert_allclose(result.mean, [4.0])
        np.testing.assert_allclose(result.var, [7.0])

    def test_addition_with_constant(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.array([1.0]), var=np.array([2.0]))
        result = g + 5.0
        np.testing.assert_allclose(result.mean, [6.0])
        np.testing.assert_allclose(result.var, [2.0])

    def test_addition_with_unsupported_type_returns_not_implemented(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.array([1.0]), var=np.array([2.0]))
        # The ufunc handler returns NotImplemented for non-numeric types -> Python falls back.
        with pytest.raises(TypeError):
            _ = g + object()  # type: ignore[operator]

    def test_eq_compares_parameters(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g1 = ArrayGaussianDistribution(mean=np.array([1.0]), var=np.array([2.0]))
        g2 = ArrayGaussianDistribution(mean=np.array([1.0]), var=np.array([2.0]))
        assert bool((g1 == g2).all())

    def test_hash_independent_from_value(self) -> None:
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        g = ArrayGaussianDistribution(mean=np.array([1.0]), var=np.array([2.0]))
        # Identity hash; just check it's an int.
        assert isinstance(hash(g), int)


class TestCreateGaussianDistribution:
    """The ``create_gaussian_distribution`` factory."""

    def test_with_var(self) -> None:
        from probly.representation.distribution import create_gaussian_distribution  # noqa: PLC0415

        g = create_gaussian_distribution(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(g.mean, [0.0, 1.0])
        np.testing.assert_allclose(g.var, [1.0, 2.0])

    def test_without_var_uses_packed_layout(self) -> None:
        from probly.representation.distribution import create_gaussian_distribution  # noqa: PLC0415

        # Last axis 2 -> [mean, var].
        packed = np.array([[1.0, 0.5], [2.0, 0.7]])
        g = create_gaussian_distribution(packed)
        np.testing.assert_allclose(g.mean, [1.0, 2.0])
        np.testing.assert_allclose(g.var, [0.5, 0.7])

    def test_without_var_wrong_packed_shape_raises(self) -> None:
        from probly.representation.distribution import create_gaussian_distribution  # noqa: PLC0415

        with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
            create_gaussian_distribution(np.array([1.0, 2.0, 3.0]))
