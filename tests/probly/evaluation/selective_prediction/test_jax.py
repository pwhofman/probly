"""JAX backend tests for selective prediction."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from probly.evaluation.selective_prediction import selective_prediction  # noqa: E402


def test_selective_prediction_exact_values() -> None:
    criterion = jnp.array([3.0, 1.0, 2.0, 0.0])
    losses = jnp.array([30.0, 10.0, 20.0, 0.0])
    aurc, bin_losses = selective_prediction(criterion, losses, n_bins=2)
    assert isinstance(aurc, jax.Array)
    assert isinstance(bin_losses, jax.Array)
    np.testing.assert_allclose(np.asarray(bin_losses), [15.0, 5.0])
    np.testing.assert_allclose(float(aurc), 10.0)


def test_selective_prediction_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    criterion = rng.random(100).astype(np.float32)
    losses = rng.random(100).astype(np.float32)
    aurc_np, bins_np = selective_prediction(criterion, losses, n_bins=10)
    aurc_jax, bins_jax = selective_prediction(jnp.asarray(criterion), jnp.asarray(losses), n_bins=10)
    np.testing.assert_allclose(np.asarray(bins_jax), bins_np, rtol=1e-5)
    np.testing.assert_allclose(float(aurc_jax), aurc_np, rtol=1e-5)


def test_selective_prediction_too_many_bins() -> None:
    with pytest.raises(ValueError, match="The number of bins can not be larger than the number of elements criterion"):
        selective_prediction(jnp.arange(5.0), jnp.arange(5.0), n_bins=10)
