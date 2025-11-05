"""Tests for flax ensemble generation."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from probly.transformation.ensemble.flax import generate_flax_ensemble


class TestModel(nnx.Module):
    """Simple test model with one Linear layer."""

    # Initializes a single Linear layer
    def __init__(self, rngs: nnx.Rngs) -> None:  # noqa: D107
        self.linear = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


def _w_b(model: TestModel) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Helper: returns weight & bias arrays for stable comparisons."""
    params = nnx.state(model)["linear"]
    w = jnp.array(params["kernel"])
    b = jnp.array(params["bias"])
    return w, b


def test_flax_ensemble_without_reset_passes() -> None:
    """Ensures ensemble members have identical random parameters (same RNG)."""
    base = TestModel(rngs=nnx.Rngs(params=0))

    # Same RNG for all
    members = [TestModel(rngs=nnx.Rngs(params=0)) for _ in range(3)]

    assert len(members) == 3
    for m in members:
        assert m is not base

    # Checks if all parameters are equal
    w0, b0 = _w_b(base)
    w1, b1 = _w_b(members[0])

    assert jnp.allclose(w0, w1)
    assert jnp.allclose(b0, b1)


def test_flax_ensemble_with_reset_passes() -> None:
    """Ensures ensemble members have different random parameters (different RNGs)."""
    base = TestModel(rngs=nnx.Rngs(params=0))
    w0, b0 = _w_b(base)

    members = generate_flax_ensemble(base, n_members=3)

    # All parameters should be different
    assert len(members) == 3
    w1, b1 = _w_b(members[0])

    assert not (jnp.allclose(w0, w1) and jnp.allclose(b0, b1))
