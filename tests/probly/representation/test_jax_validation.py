"""Eager and compiled value validation for JAX representations."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
from jax.experimental import checkify
import jax.numpy as jnp

from probly.representation.conformal_set.jax import JaxArrayOneHotConformalSet
from probly.representation.distribution.jax_bernoulli import JaxProbabilityBernoulliDistribution
from probly.representation.distribution.jax_categorical import JaxProbabilityCategoricalDistribution
from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution
from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution
from probly.representation.jax_functions import jax_mean


@pytest.mark.parametrize(
    ("constructor", "valid", "invalid", "message"),
    [
        (JaxProbabilityCategoricalDistribution, [0.2, 0.8], [-0.2, 0.8], "non-negative"),
        (JaxProbabilityBernoulliDistribution, [0.0, 1.0], [-0.1, 1.0], r"in \[0, 1\]"),
        (JaxProbabilityBernoulliDistribution, [0.0, 1.0], [0.0, 1.1], r"in \[0, 1\]"),
        (JaxDirichletDistribution, [1.0, 2.0], [0.0, 2.0], "strictly positive"),
        (
            lambda var: JaxGaussianDistribution(jnp.zeros_like(var), var),
            [1.0, 2.0],
            [0.0, 2.0],
            "Variance must be positive",
        ),
        (JaxArrayOneHotConformalSet, [1, 1, 0], [2, 1, 0], "one-hot encoded"),
        (JaxArrayOneHotConformalSet, [1, 1, 0], [-1, 1, 0], "one-hot encoded"),
    ],
)
def test_constructor_checks_eagerly_and_on_reused_compiled_calls(constructor, valid, invalid, message):
    valid = jnp.asarray(valid)
    invalid = jnp.asarray(invalid)
    expected = constructor(valid)
    with pytest.raises(ValueError, match=message):
        constructor(invalid)

    checked = jax.jit(checkify.checkify(constructor))
    for values, fails in [(valid, False), (invalid, True), (valid, False)]:
        error, result = checked(values)
        if fails:
            with pytest.raises(ValueError, match=message):
                error.throw()
        else:
            error.throw()
            assert type(result) is type(expected)
            for actual, wanted in zip(jax.tree.leaves(result), jax.tree.leaves(expected), strict=True):
                assert jnp.array_equal(actual, wanted)


def test_categorical_reconstruction_and_gradient():
    def mean_probability(values):
        distribution = JaxProbabilityCategoricalDistribution(values)
        reshaped = distribution.reshape((2, 1))
        return jax_mean(reshaped[:, 0], axis=0).probabilities[0]

    values = jnp.array([[0.2, 0.8], [0.4, 0.6]])
    error, result = jax.jit(checkify.checkify(mean_probability))(values)
    error.throw()
    assert jnp.allclose(result, 0.3)

    error, gradient = jax.jit(checkify.checkify(jax.grad(mean_probability)))(values)
    error.throw()
    assert jnp.allclose(gradient, jnp.array([[0.4, -0.1], [0.3, -0.2]]))


def test_checkified_vmap_reports_invalid_member():
    checked = jax.jit(checkify.checkify(jax.vmap(JaxProbabilityBernoulliDistribution)))
    error, result = checked(jnp.array([[0.2, 0.8], [0.4, 0.6]]))
    error.throw()
    assert result.array.shape == (2, 2)

    error, _ = checked(jnp.array([[0.2, 0.8], [0.4, 1.1]]))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        error.throw()


def test_dirichlet_sampling_validates_traced_probabilities():
    distribution = JaxDirichletDistribution(jnp.array([2.0, 3.0]))
    checked = jax.jit(checkify.checkify(lambda d, key: d.sample(8, prng_key=key)))
    error, sample = checked(distribution, jax.random.key(0))
    error.throw()
    assert sample.sample_axis == 0
    assert sample.array.probabilities.shape == (8, 2)
    assert jnp.allclose(sample.array.probabilities.sum(axis=-1), 1.0)
