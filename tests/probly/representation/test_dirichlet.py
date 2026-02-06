"""Tests for Numpy-based Dirichlet distribution representation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from probly.representation.distribution.array_dirichlet import ArrayDirichlet
from probly.representation.sampling import ArraySample


def test_array_dirichlet_initialization_valid() -> None:
    """Test standard initialization with valid numpy arrays."""
    alphas = np.array([0.5, 1.0, 2.5], dtype=float)

    dist = ArrayDirichlet(alphas=alphas)

    np.testing.assert_array_equal(dist.alphas, alphas)

    assert dist.alphas.shape == (3,)
    assert dist.shape == ()
    assert dist.ndim == 0
    assert dist.size == 1


def test_from_array_basic() -> None:
    """Test the from_array factory method."""
    alphas_list = [1, 2, 3]

    dist = ArrayDirichlet.from_array(alphas_list, dtype=np.float32)

    assert isinstance(dist, ArrayDirichlet)
    assert isinstance(dist.alphas, np.ndarray)

    assert dist.alphas.dtype == np.float32
    np.testing.assert_array_equal(dist.alphas, np.array(alphas_list, dtype=np.float32))


def test_array_dirichlet_raises_on_non_ndarray() -> None:
    """Test that __post_init__ enforces alphas to be a numpy ndarray."""
    with pytest.raises(TypeError, match="alphas must be a numpy ndarray"):
        ArrayDirichlet(alphas=[1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_array_dirichlet_raises_on_0d_array() -> None:
    """Test that alphas must have at least one dimension."""
    with pytest.raises(ValueError, match="alphas must have at least one dimension"):
        ArrayDirichlet(alphas=np.asarray(1.0))


@pytest.mark.parametrize("invalid_value", [0.0, -0.1, -5.0])
def test_array_dirichlet_raises_on_non_positive_alphas(invalid_value: float) -> None:
    """Test that concentration parameters must be strictly positive."""
    alphas = np.array([1.0, invalid_value, 2.0], dtype=float)

    with pytest.raises(ValueError, match="alphas must be strictly positive"):
        ArrayDirichlet(alphas=alphas)


def test_array_dirichlet_raises_on_too_few_classes() -> None:
    """Test that Dirichlet needs at least 2 classes (K >= 2)."""
    alphas = np.array([1.0], dtype=float)

    with pytest.raises(ValueError, match="Dirichlet distribution requires at least 2 classes"):
        ArrayDirichlet(alphas=alphas)


def test_array_properties_batched() -> None:
    """Test shape, ndim, size delegation."""
    alphas = np.ones((2, 3, 4), dtype=float)
    dist = ArrayDirichlet(alphas=alphas)

    assert dist.shape == (2, 3)
    assert dist.ndim == 2
    assert dist.size == 6


def test_len_behaviour() -> None:
    """Test __len__ behaviour (like "unsized vs sized" containers)."""
    dist_single = ArrayDirichlet.from_array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="len\\(\\) of unsized distribution"):
        _ = len(dist_single)

    dist_batch = ArrayDirichlet.from_array(np.ones((5, 3)))
    assert len(dist_batch) == 5


def test_transpose_property() -> None:
    """Test the .T property."""
    alphas = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    dist = ArrayDirichlet(alphas=alphas)

    transposed = dist.T

    # akzeptiere: entweder Objekt mit .alphas oder direkt ein Array
    got = transposed.alphas if hasattr(transposed, "alphas") else np.asarray(transposed)
    np.testing.assert_array_equal(got, alphas.T)


def test_operator_add_returns_distribution_and_keeps_positive() -> None:
    """Minimal operator/ufunc interop test."""
    dist = ArrayDirichlet.from_array([[1.0, 2.0, 3.0]])

    out = dist + 1.0
    assert isinstance(out, ArrayDirichlet)
    np.testing.assert_array_equal(out.alphas, dist.alphas + 1.0)

    out2 = dist - 1000.0
    assert isinstance(out2, ArrayDirichlet)
    assert np.all(out2.alphas >= 1e-10)


def test_copy_creates_independent_array() -> None:
    """Test copy() behaviour."""
    dist = ArrayDirichlet.from_array([[1.0, 2.0, 3.0]])
    copied = dist.copy()

    assert isinstance(copied, ArrayDirichlet)
    assert copied is not dist
    assert copied.alphas is not dist.alphas
    np.testing.assert_array_equal(copied.alphas, dist.alphas)

    copied.alphas[0, 0] = 999.0
    assert dist.alphas[0, 0] != 999.0


def test_entropy_matches_scipy_single() -> None:
    """Test entropy correctness against SciPy for a single distribution."""
    alpha = np.array([2.0, 3.0, 4.0], dtype=float)
    dist = ArrayDirichlet(alpha)

    expected = stats.dirichlet(alpha).entropy()
    assert np.all(np.isfinite(expected))
    assert np.allclose(dist.entropy, expected, rtol=1e-10, atol=1e-12)


def test_entropy_matches_scipy_batched() -> None:
    """Test entropy correctness for a batch of distributions."""
    alphas = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=float,
    )
    dist = ArrayDirichlet(alphas=alphas)

    ent = dist.entropy
    assert isinstance(ent, np.ndarray)
    assert ent.shape == (3,)
    assert np.all(np.isfinite(ent))

    expected = np.array([stats.dirichlet(a).entropy() for a in alphas], dtype=float)
    assert np.allclose(ent, expected, rtol=1e-10, atol=1e-12)


def test_sample_function_dirichlet() -> None:
    """Test the sampling function returns."""
    shape = (2, 3)
    dist = ArrayDirichlet(np.ones(shape))

    n_samples = 4
    samples = dist.num_sample(n_samples)

    assert isinstance(samples, ArraySample)
    assert samples.array.shape == (n_samples, *shape)
    assert samples.sample_axis == 0


def test_sample_simplex_constraints() -> None:
    """Samples must be valid probability vectors."""
    dist = ArrayDirichlet(np.array([1.0, 2.0, 3.0]))

    n_samples = 5000
    sample_wrapper = dist.num_sample(n_samples)
    samples = sample_wrapper.array

    assert np.all(samples >= 0.0)
    np.testing.assert_allclose(samples.sum(axis=-1), 1.0, atol=1e-12)


def test_sample_statistics_dirichlet_var() -> None:
    """Check if sample variance roughly matches Dirichlet element-wise variance."""
    alphas = np.array([2.0, 3.0, 5.0], dtype=float)
    dist = ArrayDirichlet(alphas)

    n_samples = 300000
    sample_wrapper = dist.num_sample(n_samples)
    samples = sample_wrapper.array

    a0 = alphas.sum()
    expected_var = (alphas * (a0 - alphas)) / (a0**2 * (a0 + 1.0))
    empirical_var = samples.var(axis=0)

    np.testing.assert_allclose(empirical_var, expected_var, atol=0.01)
