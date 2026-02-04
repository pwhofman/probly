"""Tests for the platt scaling implementation with flax."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from probly.calibration.scaling.flax_platt import FlaxPlatt


def test_forward(flax_setup_binary: nnx.Module) -> None:
    base, inputs, _ = flax_setup_binary
    platt_model = FlaxPlatt(base)

    platt_model.w = nnx.Param(jnp.array([0.5], dtype=jnp.float32))
    platt_model.b = nnx.Param(jnp.array([1.0], dtype=jnp.float32))

    logits_base = base(inputs)
    logits_expected = logits_base * platt_model.w + platt_model.b
    logits_scaled = platt_model(inputs)

    assert platt_model.w.shape == (1,)
    assert platt_model.b.shape == (1,)
    assert jnp.allclose(logits_scaled, logits_expected, atol=1e-5)


def test_fit(flax_setup_binary: nnx.Module) -> None:
    base, inputs, labels = flax_setup_binary
    platt_model = FlaxPlatt(base)

    dataloader = [(inputs, labels)]
    w_unoptimized = platt_model.w.clone()
    b_unoptimized = platt_model.b.clone()

    platt_model.fit(dataloader, learning_rate=0.01, max_iter=50)

    assert not jnp.allclose(platt_model.w, w_unoptimized)
    assert not jnp.allclose(platt_model.b, b_unoptimized)


def test_predict(flax_setup_binary: nnx.Module) -> None:
    base, inputs, _ = flax_setup_binary
    platt_model = FlaxPlatt(base)

    predictions = platt_model.predict(inputs)

    assert predictions.shape == (20, 1)
    assert jnp.all(predictions >= 0)
    assert jnp.all(predictions <= 1)
