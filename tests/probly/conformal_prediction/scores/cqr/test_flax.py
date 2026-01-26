"""Tests for the flax.py cqr."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.scores.cqr.common import cqr_score_func

pytest.importorskip("flax")
pytest.importorskip("jax")

import jax
from jax import Array
import jax.numpy as jnp

from probly.conformal_prediction.scores.cqr.flax import cqr_score_jax


def _cqr_score_numpy(y_true: npt.NDArray[np.floating], y_pred: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Reference NumPy implementation of the CQR nonconformity score."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float)

    if pred.ndim != 2 or pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {pred.shape}"
        raise ValueError(msg)

    lower = pred[:, 0]
    upper = pred[:, 1]

    diff_lower = lower - y
    diff_upper = y - upper
    zeros = np.zeros_like(diff_lower)

    scores = np.maximum.reduce((diff_lower, diff_upper, zeros))
    return scores.astype(float)  # type: ignore[no-any-return]


def test_cqr_score_func_numpy_matches_reference() -> None:
    """cqr_score_func with NumPy inputs should match reference implementation."""
    y_true = np.array([0.5, 1.0, -0.5, 0.0], dtype=float)
    y_pred = np.array(
        [
            [0.4, 0.6],
            [0.8, 1.2],
            [-0.6, -0.4],
            [-0.1, 0.1],
        ],
        dtype=float,
    )

    ref_scores = _cqr_score_numpy(y_true, y_pred)
    # FIX: Typ-Annotation hinzufügen
    scores: npt.NDArray[np.floating] = cqr_score_func(y_true, y_pred)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (4,)
    np.testing.assert_allclose(scores, ref_scores, rtol=1e-6, atol=1e-6)


def test_cqr_score_jax_matches_numpy_reference() -> None:
    """JAX implementation should agree with NumPy reference on same data."""
    y_true_np = np.linspace(-1.0, 1.0, num=10, dtype=float)
    lower_np = y_true_np - 0.2
    upper_np = y_true_np + 0.3
    y_pred_np = np.stack((lower_np, upper_np), axis=1)

    y_true_jax: Array = jnp.array(y_true_np, dtype=jnp.float32)
    y_pred_jax: Array = jnp.array(y_pred_np, dtype=jnp.float32)

    ref_scores = _cqr_score_numpy(y_true_np, y_pred_np)
    scores_jax = np.asarray(cqr_score_jax(y_true_jax, y_pred_jax))

    assert scores_jax.shape == ref_scores.shape
    np.testing.assert_allclose(scores_jax, ref_scores, rtol=1e-6, atol=1e-6)


def test_cqr_score_jax_with_jit_and_vmap() -> None:
    """Verify that cqr_score_jax works correctly with jax.jit and jax.vmap."""
    # simple synthetic data
    y_true_np = np.array([0.0, 0.5, 1.0, -0.5], dtype=float)
    y_pred_np = np.array(
        [
            [-0.1, 0.2],
            [0.3, 0.8],
            [0.7, 1.3],
            [-0.8, -0.2],
        ],
        dtype=float,
    )

    y_true_jax: Array = jnp.array(y_true_np, dtype=jnp.float32)
    y_pred_jax: Array = jnp.array(y_pred_np, dtype=jnp.float32)

    # base computation
    base_scores = cqr_score_jax(y_true_jax, y_pred_jax)

    # jit-compiled version
    jit_fn = jax.jit(cqr_score_jax)
    jit_scores = jit_fn(y_true_jax, y_pred_jax)

    np.testing.assert_allclose(np.asarray(jit_scores), np.asarray(base_scores), rtol=1e-6, atol=1e-6)

    # vmap over individual samples using a scalar wrapper
    def single_sample_score(y: Array, pred: Array) -> Array:
        # use leading dimension of size 1 to reuse vectorized implementation
        return cqr_score_jax(y[None], pred[None])[0]  # type: ignore[no-any-return]

    vmap_fn = jax.vmap(single_sample_score, in_axes=(0, 0))
    vmap_scores = vmap_fn(y_true_jax, y_pred_jax)

    np.testing.assert_allclose(np.asarray(vmap_scores), np.asarray(base_scores), rtol=1e-6, atol=1e-6)


def test_cqr_score_func_dispatches_to_jax_backend() -> None:
    """cqr_score_func with JAX arrays should use the JAX backend implementation."""
    y_true_np = np.array([0.2, -0.3, 0.7], dtype=float)
    y_pred_np = np.array(
        [
            [0.0, 0.5],
            [-0.5, 0.0],
            [0.4, 1.0],
        ],
        dtype=float,
    )

    y_true_jax: Array = jnp.array(y_true_np, dtype=jnp.float32)
    y_pred_jax: Array = jnp.array(y_pred_np, dtype=jnp.float32)

    # direct JAX backend
    scores_jax_direct = cqr_score_jax(y_true_jax, y_pred_jax)

    # through lazy-dispatch generic function
    # FIX: Typ-Annotation hinzufügen
    scores_generic: Array = cqr_score_func(y_true_jax, y_pred_jax)

    assert isinstance(scores_generic, jnp.ndarray)
    np.testing.assert_allclose(
        np.asarray(scores_generic),
        np.asarray(scores_jax_direct),
        rtol=1e-6,
        atol=1e-6,
    )
