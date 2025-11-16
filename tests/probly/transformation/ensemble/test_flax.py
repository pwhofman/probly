"""Tests for flax ensemble generation."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402

jax = pytest.importorskip("jax")
from jax import numpy as jnp  # noqa: E402

from probly.transformation.ensemble.flax import generate_flax_ensemble  # noqa: E402


def _w_b(layer: nnx.Module) -> tuple[jnp.array, jnp.array]:
    state = nnx.state(layer)
    w = jnp.array(state["kernel"])
    b = jnp.array(state["bias"])
    return w, b


def test_flax_ensemble_without_reset_passes(flax_model_small_2d_2d: nnx.Sequential) -> None:
    """Ensures ensemble members have identical random parameters (same RNG)."""
    # Same RNG for all
    model = generate_flax_ensemble(flax_model_small_2d_2d, n_members=3, reset_params=False)

    assert len(model) == 3
    for m in model:
        assert m is not flax_model_small_2d_2d

    # Checks if all parameters are equal
    w0, b0 = _w_b(flax_model_small_2d_2d.layers[0])
    w1, b1 = _w_b(model[0].layers[0])

    assert jnp.allclose(w0, w1)
    assert jnp.allclose(b0, b1)


def test_flax_ensemble_with_reset_passes(flax_model_small_2d_2d: nnx.Sequential) -> None:
    """Ensures ensemble members have different random parameters (different RNGs)."""
    model = generate_flax_ensemble(flax_model_small_2d_2d, n_members=3, reset_params=True)
    w0, b0 = _w_b(flax_model_small_2d_2d.layers[0])

    # All parameters should be different
    assert len(model) == 3
    w1, b1 = _w_b(model[0].layers[0])

    assert not (jnp.allclose(w0, w1) and jnp.allclose(b0, b1))
