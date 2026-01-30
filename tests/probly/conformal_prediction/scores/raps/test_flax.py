"""Tests for Flax/Jax RAPS implementation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.scores.raps.common import raps_score_func
from probly.conformal_prediction.scores.raps.flax import raps_score_jax


def test_raps_score_jax_basic() -> None:
    """Test raps_score_jax with basic data."""
    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ]
    )

    scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (2, 3)
    assert jnp.all(scores >= 0)
    # RAPS adds regularization, so can exceed 1
    assert scores[0, 0] > 0


def test_raps_score_jax_consistency_with_numpy() -> None:
    """Test that JAX implementation produces similar results to NumPy."""
    # Create test data
    np_probs = np.array(
        [
            [0.4, 0.3, 0.3],
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1],
        ]
    )
    jax_probs = jnp.array(np_probs)

    # Get scores from both implementations
    np_scores: npt.NDArray[np.floating] = raps_score_func(np_probs, lambda_reg=0.1, k_reg=0)
    jax_scores = raps_score_jax(jax_probs, lambda_reg=0.1, k_reg=0)

    # Convert JAX to numpy for comparison
    jax_scores_np = np.array(jax_scores)

    # Check shape and approximate equality
    assert np_scores.shape == jax_scores_np.shape
    assert np.allclose(np_scores, jax_scores_np, rtol=1e-5, atol=1e-5)


def test_raps_score_jax_different_lambda() -> None:
    """Test raps_score_jax with different regularization parameters."""
    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ]
    )

    scores_lambda_01 = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)
    scores_lambda_10 = raps_score_jax(probs, lambda_reg=1.0, k_reg=0)

    # Higher lambda should produce higher scores due to larger penalty
    assert jnp.all(scores_lambda_10 >= scores_lambda_01)
    assert scores_lambda_01.shape == (2, 3)
    assert scores_lambda_10.shape == (2, 3)


def test_raps_score_jax_different_k_reg() -> None:
    """Test raps_score_jax with different k_reg values."""
    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ]
    )

    scores_k0 = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)
    scores_k1 = raps_score_jax(probs, lambda_reg=0.1, k_reg=1)
    scores_k2 = raps_score_jax(probs, lambda_reg=0.1, k_reg=2)

    # With higher k_reg, penalty starts later, so scores should be lower
    assert jnp.all(scores_k1 <= scores_k0)  # k_reg=1 reduces penalty compared to k_reg=0
    assert jnp.all(scores_k2 <= scores_k1)  # k_reg=2 further reduces penalty


def test_raps_score_jax_with_epsilon() -> None:
    """Test raps_score_jax with epsilon parameter."""
    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ]
    )

    scores_no_eps = raps_score_jax(probs, lambda_reg=0.1, k_reg=0, epsilon=0.0)
    scores_with_eps = raps_score_jax(probs, lambda_reg=0.1, k_reg=0, epsilon=0.01)

    # With epsilon, all scores should be higher
    assert jnp.all(scores_with_eps >= scores_no_eps)
    # Difference should be exactly epsilon
    diff = scores_with_eps - scores_no_eps
    assert jnp.allclose(diff, 0.01, rtol=1e-5)


def test_raps_score_jax_jit_compatible() -> None:
    """Test that raps_score_jax is JIT compatible."""
    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ]
    )

    # JIT compile the function
    jitted_func = jax.jit(raps_score_jax, static_argnums=(1, 2, 3))

    # Run both regular and jitted versions
    regular_scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)
    jitted_scores = jitted_func(probs, lambda_reg=0.1, k_reg=0, epsilon=0.01)

    # Results should be identical
    assert jnp.allclose(regular_scores, jitted_scores, rtol=1e-5, atol=1e-5)


def test_raps_score_jax_gradients() -> None:
    """Test that raps_score_jax supports gradient computation."""

    def loss_fn(probs: jnp.ndarray) -> jnp.ndarray:
        scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)
        return jnp.sum(scores)

    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ]
    )

    # Compute gradient - should not raise an error
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(probs)

    assert gradients.shape == probs.shape
    # Gradients should not be all zeros (function should be differentiable)
    assert not jnp.allclose(gradients, 0.0, atol=1e-5)


def test_raps_score_jax_batch_independence() -> None:
    """Test that scores are computed independently per sample."""
    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
            [0.3, 0.3, 0.4],
        ]
    )

    # Compute scores for all samples
    all_scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)

    # Compute scores for each sample individually
    individual_scores = []
    for i in range(probs.shape[0]):
        single_probs = probs[i : i + 1, :]  # Keep batch dimension
        single_scores = raps_score_jax(single_probs, lambda_reg=0.1, k_reg=0)
        individual_scores.append(single_scores[0])  # Remove batch dimension

    # Stack individual scores
    stacked_scores = jnp.stack(individual_scores, axis=0)

    # Results should be identical
    assert jnp.allclose(all_scores, stacked_scores, rtol=1e-5, atol=1e-5)


def test_raps_score_jax_edge_case_single_sample() -> None:
    """Test raps_score_jax with single sample."""
    probs = jnp.array([[0.5, 0.3, 0.2]])
    scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (1, 3)
    assert jnp.all(scores >= 0)


def test_raps_score_jax_edge_case_single_class() -> None:
    """Test raps_score_jax with single class (should work but trivial)."""
    probs = jnp.array([[1.0]])
    scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (1, 1)
    # With one class, score should be prob + epsilon (penalty is 0 when k_reg=0)
    expected = 1.0 + 0.01  # prob + epsilon
    assert jnp.allclose(scores[0, 0], expected, rtol=1e-5)


def test_raps_score_jax_large_batch() -> None:
    """Test raps_score_jax with large batch size."""
    rng = np.random.default_rng(42)
    np_probs = rng.dirichlet([1, 1, 1], size=1000).astype(np.float32)
    probs = jnp.array(np_probs)

    scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (1000, 3)
    assert jnp.all(scores >= 0)


def test_raps_score_jax_monotonic_with_probability() -> None:
    """Test that scores are monotonic with respect to probability ordering."""
    probs = jnp.array([[0.6, 0.25, 0.15]])
    scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)

    # Get indices sorted by probability (descending)
    sorted_indices = jnp.argsort(-probs[0])

    # Get scores in probability order
    sorted_scores = scores[0, sorted_indices]
    assert jnp.all(sorted_scores[:-1] <= sorted_scores[1:])

    # highest probability class should have lowest score
    assert scores[0, 0] <= scores[0, 1]  # 0.6 vs 0.25
    assert scores[0, 0] <= scores[0, 2]  # 0.6 vs 0.15


def test_raps_score_jax_device_agnostic() -> None:
    """Test that raps_score_jax works on different devices (CPU/GPU)."""
    probs = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ]
    )

    # This should work regardless of available devices
    scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (2, 3)
    assert jnp.all(scores >= 0)


def test_raps_score_jax_dtype_preservation() -> None:
    """Test that raps_score_jax preserves input dtype."""
    # Test with float32
    probs_f32 = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
        dtype=jnp.float32,
    )

    scores_f32 = raps_score_jax(probs_f32, lambda_reg=0.1, k_reg=0)
    assert scores_f32.dtype == jnp.float32


def _iris_like_probs_jax() -> jnp.ndarray:
    """Small iris-like batch (5x4) mapped to 3-class softmax probs (5x3)."""
    x = jnp.array(
        [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3.0, 5.1, 1.8],
            [6.0, 2.2, 4.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    w = jnp.array(
        [
            [0.20, -0.10, 0.05],
            [0.10, 0.15, -0.05],
            [-0.25, 0.30, 0.10],
            [-0.10, 0.25, 0.20],
        ],
        dtype=jnp.float32,
    )
    b = jnp.array([0.10, -0.05, 0.00], dtype=jnp.float32)
    logits = x @ w + b

    # stable softmax
    logits = logits - jnp.max(logits, axis=1, keepdims=True)
    exp = jnp.exp(logits)
    return exp / jnp.sum(exp, axis=1, keepdims=True)


def test_raps_flax_iris_like_forward_shape_and_type() -> None:
    probs = _iris_like_probs_jax()
    scores = raps_score_jax(probs, lambda_reg=0.1, k_reg=1, epsilon=0.01)

    assert scores.shape == probs.shape
    assert isinstance(scores, jnp.ndarray)
    assert jnp.all(jnp.isfinite(scores))
    assert jnp.all(scores >= 0)


def test_raps_flax_iris_like_dispatch_matches_backend() -> None:
    """raps_score_func should dispatch to JAX backend and match raps_score_jax."""
    probs = _iris_like_probs_jax()

    s_dispatch = raps_score_func(probs, lambda_reg=0.1, k_reg=1, epsilon=0.01)
    s_backend = raps_score_jax(probs, lambda_reg=0.1, k_reg=1, epsilon=0.01)

    assert isinstance(s_dispatch, jnp.ndarray)
    assert s_dispatch.shape == probs.shape
    assert jnp.allclose(s_dispatch, s_backend, rtol=1e-6, atol=1e-6)


def test_raps_flax_iris_like_consistency_with_numpy() -> None:
    """JAX scores should be numerically close to NumPy implementation."""
    probs = _iris_like_probs_jax()
    probs_np = np.asarray(probs)

    s_np = raps_score_func(probs_np, lambda_reg=0.1, k_reg=1, epsilon=0.01)
    s_jax = raps_score_jax(probs, lambda_reg=0.1, k_reg=1, epsilon=0.01)

    assert np.allclose(np.asarray(s_jax), s_np, rtol=1e-5, atol=1e-6)
