"""Tests for Jax-based Dirichlet distribution representation."""

from __future__ import annotations

import pytest

from probly.representation.distribution.jax_categorical import JaxProbabilityCategoricalDistribution

pytest.importorskip("jax")
import jax
from jax import numpy as jnp
from scipy import stats

from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution
from probly.representation.jax_functions import (
    jax_average,
    jax_expand_dims,
    jax_mean,
    jax_reshape,
    jax_sum,
    jax_transpose,
)
from probly.representation.sample.jax import JaxArraySample


def test_jax_dirichlet_initialization_valid() -> None:
    """Test standard initialization with valid numpy arrays."""
    alphas = jnp.array([0.5, 1.0, 2.5], dtype=float)

    dist = JaxDirichletDistribution(alphas=alphas)

    assert jnp.equal(dist.alphas, alphas).all()

    assert dist.alphas.shape == (3,)
    assert dist.shape == ()
    assert dist.ndim == 0
    assert dist.size() == 1


def test_from_array_basic() -> None:
    """Test the from_array factory method."""
    alphas_list = [1, 2, 3]

    dist = JaxDirichletDistribution.from_array(alphas_list, dtype=jnp.float32)

    assert isinstance(dist, JaxDirichletDistribution)
    assert isinstance(dist.alphas, jax.Array)

    assert dist.alphas.dtype == jnp.float32
    assert jnp.equal(dist.alphas, jnp.array(alphas_list, dtype=jnp.float32)).all()


def test_jax_dirichlet_raises_on_non_jax_array() -> None:
    """Test that __post_init__ enforces alphas to be a jax array."""
    with pytest.raises(TypeError, match="alphas must be a jax Array"):
        JaxDirichletDistribution(alphas=[1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_jax_dirichlet_raises_on_0d_array() -> None:
    """Test that alphas must have at least one dimension."""
    with pytest.raises(ValueError, match="alphas must have at least one dimension"):
        JaxDirichletDistribution(alphas=jnp.asarray(1.0))


@pytest.mark.parametrize("invalid_value", [0.0, -0.1, -5.0])
def test_jax_dirichlet_raises_on_non_positive_alphas(invalid_value: float) -> None:
    """Test that concentration parameters must be strictly positive."""
    alphas = jnp.array([1.0, invalid_value, 2.0], dtype=float)

    with pytest.raises(ValueError, match="alphas must be strictly positive"):
        JaxDirichletDistribution(alphas=alphas)


def test_jax_dirichlet_raises_on_too_few_classes() -> None:
    """Test that Dirichlet needs at least 2 classes (K >= 2)."""
    alphas = jnp.array([1.0], dtype=float)

    with pytest.raises(ValueError, match="Dirichlet distribution requires at least 2 classes"):
        JaxDirichletDistribution(alphas=alphas)


def test_jax_properties_batched() -> None:
    """Test shape, ndim, size delegation."""
    alphas = jnp.ones((2, 3, 4), dtype=float)
    dist = JaxDirichletDistribution(alphas=alphas)

    assert dist.shape == (2, 3)
    assert dist.ndim == 2
    assert dist.size() == 6


def test_len_behaviour() -> None:
    """Test __len__ behaviour (like "unsized bs. sized containers)."""
    dist_single = JaxDirichletDistribution.from_array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match=r"len\(\) of unsized distribution"):
        _ = len(dist_single)

    dist_batch = JaxDirichletDistribution.from_array(jnp.ones((5, 3)))
    assert len(dist_batch) == 5


def test_transpose_property() -> None:
    """Test the .T property."""
    alphas = jnp.arange(24, dtype=float).reshape((2, 4, 3)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    transposed = dist.T

    assert isinstance(transposed, JaxDirichletDistribution)
    assert jnp.equal(transposed.alphas, jax_transpose(alphas, (1, 0, 2))).all()


def test_operator_add_returns_distribution_and_keeps_positive() -> None:
    """Minimal operator/ufunc interop test."""
    dist = JaxDirichletDistribution.from_array([[1.0, 2.0, 3.0]])
    out = dist + 1.0
    assert isinstance(out, JaxDirichletDistribution)
    assert jnp.equal(out.alphas, dist.alphas + 1.0).all()

    out2 = dist - 1000.0
    assert isinstance(out2, JaxDirichletDistribution)
    assert jnp.all(out2.alphas >= 1e-10)


def test_copy_creates_independent_array() -> None:
    """Test copy() behaviour."""
    dist = JaxDirichletDistribution.from_array([[1.0, 2.0, 3.0]])
    copied = dist.copy()

    assert isinstance(copied, JaxDirichletDistribution)
    assert copied is not dist
    assert copied.alphas is not dist.alphas
    assert jnp.equal(copied.alphas, dist.alphas).all()

    # jax arrays are immutable, so `.at[].set()` returns a new array rather than mutating in place.
    updated = copied.alphas.at[0, 0].set(999.0)
    assert updated[0, 0] == 999.0
    assert copied.alphas[0, 0] != 999.0
    assert dist.alphas[0, 0] != 999.0


def test_entropy_matches_scipy_single() -> None:
    """Test entropy correctness against SciPy for a single distribution."""
    alpha = jnp.array([2.0, 3.0, 4.0], dtype=float)
    dist = JaxDirichletDistribution(alpha)

    expected = stats.dirichlet(alpha).entropy()
    assert jnp.all(jnp.isfinite(expected))
    assert jnp.allclose(dist.entropy(), expected, rtol=1e-7, atol=1e-7)


def test_entropy_matches_scipy_batched() -> None:
    """Test entropy correctness for a batch of distributions."""
    alphas = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=float,
    )
    dist = JaxDirichletDistribution(alphas=alphas)

    ent = dist.entropy()
    assert isinstance(ent, jax.Array)
    assert ent.shape == (3,)
    assert jnp.all(jnp.isfinite(ent))

    expected = jnp.array([stats.dirichlet(a).entropy() for a in alphas], dtype=float)
    assert jnp.allclose(ent, expected, rtol=1e-7, atol=1e-7)


def test_sample_function_dirichlet() -> None:
    """Test the sampling function returns."""
    shape = (2, 3)
    dist = JaxDirichletDistribution(jnp.ones(shape))

    n_samples = 4
    samples = dist.sample(n_samples)

    assert isinstance(samples, JaxArraySample)
    assert samples.array.shape == (n_samples, 2)
    assert samples.sample_axis == 0


def test_sample_simplex_constraints() -> None:
    """Samples must be valid probability vectors."""
    dist = JaxDirichletDistribution(jnp.array([1.0, 2.0, 3.0]))

    n_samples = 5000
    sample_wrapper = dist.sample(n_samples)
    samples = sample_wrapper.array.probabilities

    assert jnp.all(samples >= 0.0)
    assert jnp.allclose(samples.sum(axis=-1), 1.0, atol=1e-7)


def test_sample_statistics_dirichlet_var() -> None:
    """Check if sample varaince roughly matches Dirichlet element-wise variance."""
    alphas = jnp.array([2.0, 3.0, 5.0], dtype=float)
    dist = JaxDirichletDistribution(alphas)

    n_samples = 300000
    sample_wrapper = dist.sample(n_samples)
    samples = sample_wrapper.array.probabilities

    a0 = alphas.sum()
    expected_var = (alphas * (10 - alphas)) / (10**2 * (a0 + 1.0))
    empirical_var = samples.var(axis=0)

    assert jnp.allclose(empirical_var, expected_var, atol=0.01)


def test_getitem_cannot_index_class_axis_directly() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    with pytest.raises(IndexError):
        _ = dist[:, :, 0]


def test_setitem_cannot_index_class_axis_directly() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    with pytest.raises(IndexError):
        dist.at[:, :, 0].set(jnp.array([1.0, 2.0, 3.0, 4.0]))


def test_expand_dims_last_inserts_before_class_axis() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    expanded = jax_expand_dims(dist, axis=-1)

    assert isinstance(expanded, JaxDirichletDistribution)
    assert expanded.shape == (2, 3, 1)
    assert expanded.alphas.shape == (2, 3, 1, 4)


def test_reshape_with_none_insers_before_class_axis() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    reshaped = jax_reshape(dist, shape=(6, 1), copy=None)

    assert isinstance(reshaped, JaxDirichletDistribution)
    assert reshaped.shape == (6, 1)
    assert reshaped.alphas.shape == (6, 1, 4)


def test_mean_preserves_distribution_type_and_class_axis() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    meaned = jax_mean(dist, axis=0)

    assert isinstance(meaned, JaxDirichletDistribution)
    assert meaned.shape == (3,)


def test_sum_preserves_distribution_type() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    summed = jax_sum(dist, axis=1)

    assert isinstance(summed, JaxDirichletDistribution)
    assert summed.shape == (2,)
    assert jnp.allclose(summed.alphas, jax_sum(alphas, axis=1))


def test_average_preserves_distribution_type_and_uses_weights() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    weights = jnp.array([0.25, 0.75])
    dist = JaxDirichletDistribution(alphas=alphas)

    averaged = jax_average(dist, axis=0, weights=weights)

    assert isinstance(averaged, JaxDirichletDistribution)
    assert averaged.shape == (3,)
    assert jnp.allclose(averaged.alphas, jax_average(alphas, axis=0, weights=weights))


def test_mean_returns_probability_categorical() -> None:
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    mean = dist.mean

    assert isinstance(mean, JaxProbabilityCategoricalDistribution)


def test_eq_with_dirchlet() -> None:
    """``__eq__`` accepts dirichlet comparison."""
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist1 = JaxDirichletDistribution(alphas=alphas)
    dist2 = JaxDirichletDistribution(alphas=alphas)

    assert isinstance(dist1, JaxDirichletDistribution)
    assert isinstance(dist2, JaxDirichletDistribution)
    assert dist1 is not dist2
    assert dist1.__eq__(dist2).all()


def test_eq_with_alphas() -> None:
    """``__eq__`` accepts non-dirichlet comparison."""
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    assert isinstance(dist, JaxDirichletDistribution)
    assert not isinstance(alphas, JaxDirichletDistribution)
    assert dist.__eq__(alphas).all()


def test_hash_returns_identity() -> None:
    """``hash`` returns int."""
    alphas = jnp.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = JaxDirichletDistribution(alphas=alphas)

    assert isinstance(dist, JaxDirichletDistribution)
    assert isinstance(hash(dist), int)
