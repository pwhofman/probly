"""Tests for JAX-based Gaussian distribution representation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution
from probly.representation.sample.jax import JaxArraySample


def test_jax_gaussian_initialization_valid() -> None:
    """Test standard initialization with valid JAX arrays as well as types."""
    mean = jnp.array([0.0, 1.0])
    var = jnp.array([1.0, 0.5])

    dist = JaxGaussianDistribution(mean=mean, var=var)

    assert jnp.allclose(dist.mean, mean)
    assert jnp.allclose(dist.var, var)
    assert dist.type == "gaussian"


def test_jax_gaussian_raises_on_shape_mismatch() -> None:
    """The constructor raises ``ValueError`` when ``mean`` and ``var`` shapes differ."""
    mean = jnp.zeros((5,))
    var = jnp.ones((4,))

    with pytest.raises(ValueError, match="mean and var must have same shape"):
        JaxGaussianDistribution(mean=mean, var=var)


@pytest.mark.parametrize("invalid_var", [0.0, -0.1, -5.0])
def test_jax_gaussian_raises_on_non_positive_variance(invalid_var: float) -> None:
    """The constructor raises ``ValueError`` when any variance entry is non-positive."""
    mean = jnp.array([0.0, 0.0])
    var = jnp.array([1.0, invalid_var])

    with pytest.raises(ValueError, match="Variance must be positive"):
        JaxGaussianDistribution(mean=mean, var=var)


def test_from_parameters_creates_instance() -> None:
    """Test the constructor with array inputs."""
    mean_array = jnp.array([1.0, 2.0])
    var_array = jnp.array([0.5, 0.5])

    dist = JaxGaussianDistribution(mean=mean_array, var=var_array)

    assert isinstance(dist, JaxGaussianDistribution)
    assert jnp.allclose(dist.mean, mean_array)
    assert jnp.allclose(dist.var, var_array)


def test_jax_properties() -> None:
    """Test shape and ndim delegation through the protected-axis base."""
    shape = (2, 3)
    mean = jnp.zeros(shape)
    var = jnp.ones(shape)

    dist = JaxGaussianDistribution(mean, var)

    assert dist.shape == shape
    assert dist.ndim == 2


def test_std() -> None:
    """Standard deviation is the elementwise square root of the variance."""
    dist = JaxGaussianDistribution(jnp.array([0.0, 1.0]), jnp.array([1.0, 4.0]))

    assert jnp.allclose(dist.std, jnp.array([1.0, 2.0]))


def test_quantile() -> None:
    """Test the Gaussian quantile calculation."""
    dist = JaxGaussianDistribution(jnp.array([0.0, 1.0]), jnp.array([1.0, 4.0]))

    scalar_quantile = dist.quantile(0.5)
    vector_quantile = dist.quantile(jnp.array([0.5, 0.8413447]))

    assert jnp.allclose(scalar_quantile, dist.mean)
    assert vector_quantile.shape == (2, 2)
    assert jnp.allclose(vector_quantile[:, 0], dist.mean)


def test_sample_function() -> None:
    """The sampling function returns a ``JaxArraySample`` with the expected shape."""
    shape = (2,)
    dist = JaxGaussianDistribution(jnp.zeros(shape), jnp.ones(shape))

    n_samples = 4
    samples = dist.sample(n_samples, rng=jax.random.key(0))

    assert isinstance(samples, JaxArraySample)
    assert samples.array.shape == (n_samples, *shape)
    assert samples.sample_axis == 0


def test_sample_statistics() -> None:
    """The empirical mean and variance match the distribution parameters."""
    mean_val = 10.0
    var_val = 4.0
    dist = JaxGaussianDistribution(jnp.array([mean_val]), jnp.array([var_val]))

    n_samples = 100_000
    sample_wrapper = dist.sample(n_samples, rng=jax.random.key(0))
    samples = sample_wrapper.array

    assert float(jnp.mean(samples)) == pytest.approx(mean_val, abs=0.1)
    assert float(jnp.var(samples)) == pytest.approx(var_val, abs=0.1)


def test_slice() -> None:
    """Slicing via ``__getitem__`` returns a new ``JaxGaussianDistribution``."""
    mean = jnp.array([10.0, 20.0, 30.0])
    var = jnp.array([1.0, 1.0, 1.0])
    dist = JaxGaussianDistribution(mean, var)

    sliced = dist[:2]

    assert isinstance(sliced, JaxGaussianDistribution)
    assert sliced.shape == (2,)
    assert jnp.allclose(sliced.mean, jnp.array([10.0, 20.0]))
    assert jnp.allclose(sliced.var, jnp.array([1.0, 1.0]))


def test_sample_with_same_key_is_deterministic() -> None:
    """Two ``sample`` calls with the same ``rng`` key return identical samples."""
    dist = JaxGaussianDistribution(jnp.zeros((2,)), jnp.ones((2,)))
    key = jax.random.key(7)

    sample_a = dist.sample(num_samples=8, rng=key)
    sample_b = dist.sample(num_samples=8, rng=key)

    assert jnp.array_equal(sample_a.array, sample_b.array)


def test_sample_with_different_keys_differs() -> None:
    """Two ``sample`` calls with distinct ``rng`` keys return distinct samples."""
    dist = JaxGaussianDistribution(jnp.zeros((2,)), jnp.ones((2,)))

    sample_a = dist.sample(num_samples=8, rng=jax.random.key(0))
    sample_b = dist.sample(num_samples=8, rng=jax.random.key(1))

    assert not jnp.array_equal(sample_a.array, sample_b.array)


def test_create_gaussian_distribution_from_jax_split_array() -> None:
    """Passing a single ``(..., 2)`` array splits into mean and variance."""
    from probly.representation.distribution import create_gaussian_distribution  # noqa: PLC0415

    stacked = jnp.stack((jnp.array([0.0, 1.0]), jnp.array([1.0, 0.5])), axis=-1)
    dist = create_gaussian_distribution(stacked)

    assert isinstance(dist, JaxGaussianDistribution)
    assert jnp.allclose(dist.mean, jnp.array([0.0, 1.0]))
    assert jnp.allclose(dist.var, jnp.array([1.0, 0.5]))
