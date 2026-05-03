"""Tests for JAX categorical distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from probly.representation.distribution import create_categorical_distribution
from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution
from probly.representation.sample.jax import JaxArraySample


def test_create_categorical_distribution_from_jax_array() -> None:
    probabilities = jnp.array([[0.2, 0.3, 0.5]], dtype=jnp.float32)

    dist = create_categorical_distribution(probabilities)

    assert isinstance(dist, JaxCategoricalDistribution)
    assert jnp.array_equal(dist.probabilities, probabilities)


def test_accepts_relative_non_negative_probabilities() -> None:
    probabilities = jnp.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)

    dist = JaxCategoricalDistribution(probabilities)

    assert dist.shape == (2,)
    assert dist.num_classes == 3


def test_rejects_negative_relative_probabilities() -> None:
    probabilities = jnp.array([1.0, -1.0, 2.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="non-negative"):
        JaxCategoricalDistribution(probabilities)


def test_entropy_normalizes_relative_probabilities() -> None:
    probabilities = jnp.array([[2.0, 3.0, 5.0]], dtype=jnp.float32)
    dist = JaxCategoricalDistribution(probabilities)

    normalized = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    expected = -jnp.sum(normalized * jnp.log(normalized), axis=-1)

    assert jnp.allclose(dist.entropy, expected)


def test_entropy_bernoulli_formula() -> None:
    probabilities = jnp.array([[0.25], [0.5], [0.75]], dtype=jnp.float32)
    dist = JaxCategoricalDistribution(probabilities)

    p = probabilities[:, 0]
    expected = -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))

    assert jnp.allclose(dist.entropy, expected)


def test_sampling_relative_probabilities_matches_normalized_distribution() -> None:
    probabilities = jnp.array([[2.0, 3.0, 5.0]], dtype=jnp.float32)
    dist = JaxCategoricalDistribution(probabilities)
    rng = jax.random.key(0)

    sample = dist.sample(num_samples=30_000, rng=rng)

    assert isinstance(sample, JaxArraySample)
    assert sample.sample_axis == 0
    assert sample.array.shape == (30_000, 1)

    counts = jnp.bincount(sample.array[:, 0], length=dist.num_classes).astype(jnp.float32)
    frequencies = counts / jnp.sum(counts)
    expected = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)

    assert jnp.allclose(frequencies, expected, atol=0.02)


def test_sampling_bernoulli_produces_binary_samples_with_correct_mean() -> None:
    probabilities = jnp.array([[0.3]], dtype=jnp.float32)
    dist = JaxCategoricalDistribution(probabilities)
    rng = jax.random.key(1)

    sample = dist.sample(num_samples=40_000, rng=rng)

    assert isinstance(sample, JaxArraySample)
    assert sample.array.shape == (40_000, 1)
    assert jnp.all((sample.array == 0) | (sample.array == 1))
    assert float(jnp.mean(sample.array.astype(jnp.float32))) == pytest.approx(0.3, abs=0.02)


def test_numpy_array_interop() -> None:
    probabilities = jnp.array([[0.2, 0.8]], dtype=jnp.float32)
    dist = JaxCategoricalDistribution(probabilities)

    array = np.asarray(dist, dtype=np.float32)

    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    np.testing.assert_allclose(array, np.array([[0.2, 0.8]], dtype=np.float32))


def test_getitem_cannot_index_class_axis_directly() -> None:
    probabilities = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4)) + 1.0
    dist = JaxCategoricalDistribution(probabilities)

    with pytest.raises(IndexError):
        _ = dist[:, :, 0]


def test_getitem_keeps_protected_class_axis() -> None:
    probabilities = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4)) + 1.0
    dist = JaxCategoricalDistribution(probabilities)

    sliced = dist[0]

    assert isinstance(sliced, JaxCategoricalDistribution)
    assert sliced.shape == (3,)
    assert sliced.unnormalized_probabilities.shape == (3, 4)


def test_sample_with_same_key_is_deterministic() -> None:
    """Two ``sample`` calls with the same ``rng`` key return identical samples."""
    probabilities = jnp.array([[0.2, 0.3, 0.5]], dtype=jnp.float32)
    dist = JaxCategoricalDistribution(probabilities)
    key = jax.random.key(7)

    sample_a = dist.sample(num_samples=8, rng=key)
    sample_b = dist.sample(num_samples=8, rng=key)

    assert jnp.array_equal(sample_a.array, sample_b.array)


def test_sample_with_different_keys_differs() -> None:
    """Two ``sample`` calls with distinct ``rng`` keys return distinct samples."""
    probabilities = jnp.array([[0.2, 0.3, 0.5]], dtype=jnp.float32)
    dist = JaxCategoricalDistribution(probabilities)

    sample_a = dist.sample(num_samples=64, rng=jax.random.key(0))
    sample_b = dist.sample(num_samples=64, rng=jax.random.key(1))

    assert not jnp.array_equal(sample_a.array, sample_b.array)
