"""Tests for Jax-based categorical distribution representation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from probly.representation.distribution import create_categorical_distribution
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.jax_functions import jax_average, jax_concatenate, jax_expand_dims, jax_mean, jax_stack
from probly.representation.sample.jax import JaxArraySample


def test_accepts_relative_non_negative_probabilities() -> None:
    probabilities = jnp.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=float)

    dist = JaxProbabilityCategoricalDistribution(probabilities)

    assert dist.shape == (2,)
    assert dist.num_classes == 3


def test_create_categorical_distribution_from_jax_array() -> None:
    probabilities = jnp.array([[2.0, 3.0, 5.0]], dtype=float)

    dist = create_categorical_distribution(probabilities)

    assert isinstance(dist, JaxCategoricalDistribution)
    assert jnp.allclose(dist.unnormalized_probabilities, probabilities)


def test_rejects_negative_relative_probabilities() -> None:
    probabilities = jnp.array([1.0, -1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        JaxProbabilityCategoricalDistribution(probabilities)


def test_zero_sum_relative_probabilities_return_nan() -> None:
    dist = JaxProbabilityCategoricalDistribution(jnp.array([0.0, 0.0, 0.0], dtype=float))

    assert jnp.isnan(dist.probabilities).all()


def test_entropy_normalizes_relative_probabilities() -> None:
    probabilities = jnp.array([[2.0, 3.0, 5.0]], dtype=float)
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    normalized = probabilities / probabilities.sum(axis=-1, keepdims=True)
    expected = -jnp.sum(normalized * jnp.log(normalized), axis=-1)

    assert jnp.allclose(dist.entropy(), expected)


def test_sampling_relative_probabilities_matches_normalized_distribution() -> None:
    probabilities = jnp.array([[2.0, 3.0, 5.0]], dtype=float)
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    sample = dist.sample(num_samples=30_000, prng_key=jax.random.key(1))

    assert isinstance(sample, JaxArraySample)
    assert sample.sample_axis == 0
    assert sample.array.shape == (30_000, 1)
    assert sample.array.dtype == jnp.int32

    values, counts = jnp.unique(sample.array[:, 0], return_counts=True)
    frequencies = jnp.zeros(dist.num_classes)
    frequencies = frequencies.at[values].set(counts / counts.sum())
    expected = jnp.array([0.2, 0.3, 0.5], dtype=float)

    assert jnp.allclose(frequencies, expected, atol=0.02)


def test_sampling_without_a_key_draws_fresh_samples() -> None:
    dist = JaxProbabilityCategoricalDistribution(jnp.array([[0.5, 0.5]], dtype=float))

    first = dist.sample(num_samples=64)
    second = dist.sample(num_samples=64)

    assert not bool(jnp.all(first.array == second.array))


def test_sampling_with_an_explicit_key_is_deterministic() -> None:
    dist = JaxProbabilityCategoricalDistribution(jnp.array([[0.5, 0.5]], dtype=float))

    first = dist.sample(num_samples=64, prng_key=jax.random.key(3))
    second = dist.sample(num_samples=64, prng_key=jax.random.key(3))

    assert bool(jnp.all(first.array == second.array))


def test_getitem_cannot_index_class_axis_directly() -> None:
    probabilities = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    with pytest.raises(IndexError):
        _ = dist[:, :, 0]


def test_setitem_cannot_index_class_axis_directly() -> None:
    probabilities = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    with pytest.raises(IndexError):
        dist.at[:, :, 0].set(jnp.array([1.0, 2.0, 3.0, 4.0]))


def test_expand_dims_last_inserts_before_class_axis() -> None:
    probabilities = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    expanded = jax_expand_dims(dist, axis=-1)

    assert isinstance(expanded, JaxCategoricalDistribution)
    assert expanded.shape == (2, 3, 1)
    assert expanded.probabilities.shape == (2, 3, 1, 4)


def test_reshape_with_none_inserts_before_class_axis() -> None:
    probabilities = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    reshaped = dist.reshape((6, None))

    assert isinstance(reshaped, JaxCategoricalDistribution)
    assert reshaped.shape == (6, 1)
    assert reshaped.probabilities.shape == (6, 1, 4)


def test_concatenate_preserves_distribution_type() -> None:
    probabilities = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    concatenated = jax_concatenate((dist, dist), axis=-1)

    assert isinstance(concatenated, JaxCategoricalDistribution)
    assert concatenated.shape == (2, 6)
    assert concatenated.probabilities.shape == (2, 6, 4)


def test_stack_preserves_distribution_type() -> None:
    probabilities = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxProbabilityCategoricalDistribution(probabilities)

    stacked = jax_stack((dist, dist), axis=0)

    assert isinstance(stacked, JaxCategoricalDistribution)
    assert stacked.shape == (2, 2, 3)
    assert stacked.probabilities.shape == (2, 2, 3, 4)


def test_mean_preserves_distribution_type_and_class_axis() -> None:
    unnormalized = jnp.array(
        [[[1.0, 1.0], [1.0, 3.0]], [[9.0, 1.0], [2.0, 2.0]]],
        dtype=float,
    )
    dist = JaxProbabilityCategoricalDistribution(unnormalized)

    meaned = jax_mean(dist, axis=0)

    assert isinstance(meaned, JaxCategoricalDistribution)
    assert meaned.shape == (2,)
    expected = jnp.mean(unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True), axis=0)
    assert jnp.allclose(meaned.probabilities, expected)


def test_average_preserves_distribution_type_and_uses_weigts() -> None:
    unnormalized = jnp.array(
        [[[1.0, 1.0], [1.0, 3.0]], [[9.0, 1.0], [2.0, 2.0]]],
        dtype=float,
    )
    weights = jnp.array([0.25, 0.75])
    dist = JaxProbabilityCategoricalDistribution(unnormalized)

    averaged = jax_average(dist, axis=0, weights=weights)

    assert isinstance(averaged, JaxCategoricalDistribution)
    assert averaged.shape == (2,)
    probabilities = unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)
    expected = jnp.average(probabilities, axis=0, weights=weights)
    assert jnp.allclose(averaged.probabilities, expected)


def test_hash_is_identity_based_and_distinguished_instances() -> None:
    probabilities = jnp.array([[0.2, 0.8]], dtype=float)
    dist_a = JaxProbabilityCategoricalDistribution(probabilities.copy())
    dist_b = JaxProbabilityCategoricalDistribution(probabilities.copy())

    assert hash(dist_a) == hash(dist_a)
    assert hash(dist_a) != hash(dist_b)


class TestArrayCategoricalDistributionPostprocessing:
    """Ensure protected-axis processing rebuilds a ProbabilityCategoricalDistribution after jnp.mean."""

    def test_mean_returns_probability_distribution(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
            JaxProbabilityCategoricalDistribution,
        )

        # A LogitCategorical reduced via jax_mean should land as a Probability distribution.
        d = JaxLogitCategoricalDistribution(array=jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
        result = jax_mean(d, axis=0)
        # The mean of a Logit-cat distribution should produce a probability distribution.
        assert isinstance(result, JaxProbabilityCategoricalDistribution)


class TestArrayCategoricalDistribution:
    """Validation, equality and sampling for the jax categorical distribution."""

    def test_negative_probabilites_raise(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxProbabilityCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="non-negative"):
            JaxProbabilityCategoricalDistribution(array=jnp.array([0.5, -0.1, 0.6]))

    def test_zero_dim_array_raises(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxProbabilityCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="at least one dimension"):
            JaxProbabilityCategoricalDistribution(array=jnp.array(0.5))

    def test_array_must_be_jax_array(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxProbabilityCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="jax Array"):
            JaxProbabilityCategoricalDistribution(array=[0.5, 0.5])  # type: ignore[arg-type]

    def test_logit_array_must_be_jax_array(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="jax array"):
            JaxLogitCategoricalDistribution(array=[0.5, 0.5])  # type: ignore[arg-type]

    def test_logit_zero_dim_raises(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="at least one dimension"):
            JaxLogitCategoricalDistribution(array=jnp.array(0.5))

    def test_eq_two_probability_distributions(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxProbabilityCategoricalDistribution,
        )

        d1 = JaxProbabilityCategoricalDistribution(array=jnp.array([[0.2, 0.3, 0.5]]))
        d2 = JaxProbabilityCategoricalDistribution(array=jnp.array([[0.4, 0.6, 1.0]]))
        # After normalization both have the same probabilites.
        assert bool(d1 == d2)

    def test_eq_with_array(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxProbabilityCategoricalDistribution,
        )

        d1 = JaxProbabilityCategoricalDistribution(array=jnp.array([[0.2, 0.3, 0.5]]))
        # Comparison with a raw array uses unnormalized probabilities.
        eq = d1 == jnp.array([[0.2, 0.3, 0.5]])
        assert bool(eq)

    def test_logit_eq(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
        )

        a = JaxLogitCategoricalDistribution(array=jnp.array([[1.0, 2.0, 3.0]]))
        b = JaxLogitCategoricalDistribution(array=jnp.array([[1.0, 2.0, 3.0]]))
        assert bool(a == b)

    def test_logit_eq_with_array(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
        )

        a = JaxLogitCategoricalDistribution(array=jnp.array([[1.0, 2.0, 3.0]]))
        eq = a == jnp.array([[1.0, 2.0, 3.0]])
        assert bool(eq)

    def test_hash(self) -> None:
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
            JaxProbabilityCategoricalDistribution,
        )

        a = JaxProbabilityCategoricalDistribution(array=jnp.array([[0.5, 0.5]]))
        b = JaxLogitCategoricalDistribution(array=jnp.array([[0.0, 1.0]]))
        # Identity-based hash returns ints.
        assert isinstance(hash(a), int)
        assert isinstance(hash(b), int)
