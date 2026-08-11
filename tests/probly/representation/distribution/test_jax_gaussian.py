"""Tests for Jax-based Gaussian distribution representation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution
from probly.representation.sample.jax import JaxArraySample


def test_jax_gaussian_initialization_valid() -> None:
    """Test standard initialization with valid jax arrays aswell as types."""
    mean = jnp.array([0.0, 1.0])
    var = jnp.array([1.0, 0.5])

    dist = JaxGaussianDistribution(mean=mean, var=var)

    assert jnp.equal(dist.mean, mean).all()
    assert jnp.equal(dist.var, var).all()
    assert dist.type == "gaussian"
    assert dist.mean.dtype == mean.dtype
    assert dist.var.dtype == var.dtype


def test_jax_gaussian_raises_on_shape_mismatch() -> None:
    """Test if the function does raise a ValueError upon wrong initalization."""
    mean = jnp.zeros((5,))
    var = jnp.ones((4,))

    with pytest.raises(ValueError, match="mean and var must have same shape"):
        JaxGaussianDistribution(mean=mean, var=var)


@pytest.mark.parametrize("invalid_var", [0.0, -0.1, -5.0])
def test_jax_gaussian_raises_on_non_positve_variance(invalid_var: float) -> None:
    """Test if the function does raise a ValueError upon using a negative variance."""
    mean = jnp.array([0.0, 0.0])
    var = jnp.array([1.0, invalid_var])

    with pytest.raises(ValueError, match="Variance must be positive"):
        JaxGaussianDistribution(mean=mean, var=var)


def test_from_parameters_creates_instance() -> None:
    """Test the from_parameters factory method."""
    mean_list = [1.0, 2.0]
    var_list = [0.5, 0.5]

    dist = JaxGaussianDistribution(mean=mean_list, var=var_list)

    assert isinstance(dist, JaxGaussianDistribution)

    assert jnp.equal(dist.mean, jnp.array(mean_list, dtype=float)).all()
    assert jnp.equal(dist.var, jnp.array(var_list, dtype=float)).all()


def test_jax_properties() -> None:
    """Test shape, ndim, size delegation."""
    shape = (2, 3)
    mean = jnp.zeros(shape)
    var = jnp.ones(shape)

    dist = JaxGaussianDistribution(mean, var)

    assert dist.shape == shape
    assert dist.ndim == 2
    assert dist.size() == 6
    assert dist.__array_namespace__() is jnp


def test_std() -> None:
    """Test standard deviation calculation."""
    dist = JaxGaussianDistribution(jnp.array([0.0, 1.0]), jnp.array([1.0, 4.0]))

    assert jnp.equal(dist.std, jnp.array([1.0, 2.0])).all()


def test_quantile() -> None:
    """Test Gaussian quantile calculation."""
    dist = JaxGaussianDistribution(jnp.array([0.0, 1.0]), jnp.array([1.0, 4.0]))

    scalar_quantile = dist.quantile(0.5)
    vector_quantile = dist.quantile(jnp.array([0.5, 0.8413447]))

    assert jnp.allclose(scalar_quantile, dist.mean)
    assert vector_quantile.shape == (2, 2)
    assert jnp.allclose(vector_quantile[:, 0], dist.mean)


def test_transpose_property() -> None:
    """Test the .T property."""
    mean = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    var = jnp.array([[0.1, 0.2], [0.3, 0.4]])

    dist = JaxGaussianDistribution(mean, var)
    transposed = dist.T

    assert isinstance(transposed, JaxGaussianDistribution)
    assert transposed.shape == (2, 2)
    assert jnp.equal(transposed.mean, mean.T).all()


def test_matrix_transpose_property() -> None:
    """Test the .mT property."""
    shape = (2, 3, 4)
    mean = jnp.zeros(shape)
    var = jnp.ones(shape)
    dist = JaxGaussianDistribution(mean, var)

    t_dist = dist.T

    assert t_dist.shape == (4, 3, 2)
    expected_mean = jnp.transpose(mean)
    assert jnp.equal(t_dist.mean, expected_mean).all()


def test_sample_function() -> None:
    """Test the sampling function returns."""
    shape = (2,)
    dist = JaxGaussianDistribution(jnp.zeros(shape), jnp.ones(shape))

    n_samples = 4
    samples = dist.sample(n_samples)

    assert isinstance(samples, JaxArraySample)
    assert samples.array.shape == (n_samples, *shape)
    assert samples.sample_axis == 0


def test_sample_statistics() -> None:
    """Check if the samples actually follow the Gaussian distribution statistically."""
    mean_val = 10.0
    var_val = 4.0
    dist = JaxGaussianDistribution(jnp.array([mean_val]), jnp.array([var_val]))

    n_samples = 100000
    sample_wrapper = dist.sample(n_samples)
    samples = sample_wrapper.array

    assert jnp.mean(samples) == pytest.approx(mean_val, abs=0.1)
    assert jnp.var(samples) == pytest.approx(var_val, abs=0.1)


def test_entropy() -> None:
    """Test if entropy calculation works properly."""
    mean = jnp.array([0])
    var = jnp.array([1])

    dist = JaxGaussianDistribution(mean=mean, var=var)

    expected = 0.5 * jnp.log(2 * jnp.pi * jnp.e * var)
    assert dist.entropy() == pytest.approx(expected)


def test_slice() -> None:
    """Test slicing via __getitem__ returns a new JaxGaussian."""
    mean = jnp.array([10.0, 20.0, 30.0])
    var = jnp.array([1.0, 1.0, 1.0])
    dist = JaxGaussianDistribution(mean, var)

    sliced = dist[:2]

    assert isinstance(sliced, JaxGaussianDistribution)
    assert sliced.shape == (2,)
    assert jnp.equal(sliced.mean, jnp.array([10.0, 20.0])).all()
    assert jnp.equal(sliced.var, jnp.array([1.0, 1.0])).all()


def test_copy_method() -> None:
    """Test copying."""
    mean = jnp.array([1.0])
    var = jnp.array([1.0])
    dist = JaxGaussianDistribution(mean, var)

    copied = dist.copy()

    assert copied == dist
    assert copied is not dist
    assert copied.mean is not dist.mean


class TestJaxGaussianDistribution:
    """Jax-based Gaussian distribution."""

    def test_mismatched_shapes_raise(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        with pytest.raises(ValueError, match="same shape"):
            JaxGaussianDistribution(mean=jnp.zeros((3,)), var=jnp.ones((4,)))

    def test_non_positive_var_raises(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        with pytest.raises(ValueError, match="must be positive"):
            JaxGaussianDistribution(mean=jnp.zeros((3,)), var=jnp.zeros((3,)))

    def test_std_property(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.array([0.0, 1.0]), var=jnp.array([4.0, 9.0]))
        assert jnp.allclose(g.std, jnp.array([2.0, 3.0]))

    def test_quantile_scalar_q(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.array([0.0]), var=jnp.array([1.0]))
        # At q=0.5, the median equals the mean.
        assert jnp.allclose(g.quantile(0.5), jnp.array([0.0]), atol=1e-6)

    def test_quantile_array_q(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.array([0.0]), var=jnp.array([1.0]))
        result = g.quantile(jnp.array([0.5, 0.5]))
        # Two queries -> two outputs each of shape (1,).
        assert result.shape == (1, 2)

    def test_sample_returns_correct_shape(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.zeros((3,)), var=jnp.ones((3,)))
        samples = g.sample(num_samples=5)
        assert samples.array.shape == (5, 3)
        assert samples.sample_axis == 0

    def test_samples_uses_prng(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.zeros((2,)), var=jnp.ones((2,)))
        prng1 = jax.random.key(42)
        prng2 = jax.random.key(42)
        s1 = g.sample(num_samples=3, prng_key=prng1)
        s2 = g.sample(num_samples=3, prng_key=prng2)
        # same seed -> same samples.
        assert jnp.allclose(s1.array, s2.array)

    def test_jax_dunder_stacks_mean_var(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.array([1.0, 2.0]), var=jnp.array([0.5, 0.5]))
        arr = jnp.asarray(g)
        # Last axis stacks mean and var.
        assert jnp.allclose(arr[..., 0], jnp.array([1.0, 2.0]))
        assert jnp.allclose(arr[..., 1], jnp.array([0.5, 0.5]))

    def test_addition_of_two_gaussians(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g1 = JaxGaussianDistribution(mean=jnp.array([1.0]), var=jnp.array([2.0]))
        g2 = JaxGaussianDistribution(mean=jnp.array([3.0]), var=jnp.array([5.0]))
        result = g1 + g2
        # Means add, variances add.
        assert jnp.allclose(result.mean, jnp.array([4.0]))
        assert jnp.allclose(result.var, jnp.array([7.0]))

    def test_addition_with_constant(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.array([1.0]), var=jnp.array([2.0]))
        result = g + 5.0
        assert jnp.allclose(result.mean, jnp.array([6.0]))
        assert jnp.allclose(result.var, jnp.array([2.0]))

    def test_addition_with_unsupported_type_returns_not_implemented(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.array([1.0]), var=jnp.array([2.0]))
        # The ufunc handler returns NotImplemented for non-numeric types -> Python falls back.
        with pytest.raises(TypeError):
            _ = g + object()  # type: ignore[operator]

    def test_eq_compares_parameters(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g1 = JaxGaussianDistribution(mean=jnp.array([1.0]), var=jnp.array([2.0]))
        g2 = JaxGaussianDistribution(mean=jnp.array([1.0]), var=jnp.array([2.0]))
        assert bool((g1 == g2).all())

    def test_hash_independent_from_value(self) -> None:
        from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution  # noqa: PLC0415

        g = JaxGaussianDistribution(mean=jnp.array([1.0]), var=jnp.array([2.0]))
        # Identity hash; just check if it's an int.
        assert isinstance(hash(g), int)


class TestCreateGaussianDistribution:
    """The ``create_gaussian_distribution`` factory."""

    def test_with_var(self) -> None:
        from probly.representation.distribution import create_gaussian_distribution  # noqa: PLC0415

        g = create_gaussian_distribution(jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0]))
        assert jnp.allclose(g.mean, jnp.array([0.0, 1.0]))
        assert jnp.allclose(g.var, jnp.array([1.0, 2.0]))

    def test_without_var_uses_packed_layout(self) -> None:
        from probly.representation.distribution import create_gaussian_distribution  # noqa: PLC0415

        # Las axis 2 -> [mean, var].
        packed = jnp.array([[1.0, 0.5], [2.0, 0.7]])
        g = create_gaussian_distribution(packed)
        assert jnp.allclose(g.mean, jnp.array([1.0, 2.0]))
        assert jnp.allclose(g.var, jnp.array([0.5, 0.7]))

    def test_without_var_wrong_packed_shape_raises(self) -> None:
        from probly.representation.distribution import create_gaussian_distribution  # noqa: PLC0415

        with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
            create_gaussian_distribution(jnp.array([1.0, 2.0, 3.0]))
