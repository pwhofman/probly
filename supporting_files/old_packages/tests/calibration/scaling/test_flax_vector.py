"""Tests for the vector scaling implementation with flax."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from probly.calibration.scaling.flax_vector import FlaxVector


def test_forward(flax_setup_multiclass: nnx.Module) -> None:
    base, inputs, _ = flax_setup_multiclass
    vector_model = FlaxVector(base, num_classes=3)

    vector_model.w = nnx.Param(jnp.array([0.5, 0.25, 0.1], dtype=jnp.float32))
    vector_model.b = nnx.Param(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))

    logits_base = base(inputs)
    logits_expected = logits_base * vector_model.w + vector_model.b
    logits_scaled = vector_model(inputs)

    assert vector_model.w.shape == (3,)
    assert vector_model.b.shape == (3,)
    assert jnp.allclose(logits_scaled, logits_expected, atol=1e-5)


def test_fit(flax_setup_multiclass: nnx.Module) -> None:
    base, inputs, labels = flax_setup_multiclass
    vector_model = FlaxVector(base, num_classes=3)

    dataloader = [(inputs, labels)]
    w_unoptimized = vector_model.w.clone()
    b_unoptimized = vector_model.b.clone()

    vector_model.fit(dataloader, learning_rate=0.01, max_iter=50)

    assert not jnp.allclose(vector_model.w, w_unoptimized)
    assert not jnp.allclose(vector_model.b, b_unoptimized)


def test_predict(flax_setup_multiclass: nnx.Module) -> None:
    base, inputs, _ = flax_setup_multiclass
    vector_model = FlaxVector(base, num_classes=3)

    predictions = vector_model.predict(inputs)

    assert predictions.shape == (20, 3)
    assert jnp.all(predictions >= 0)
    assert jnp.all(predictions <= 1)
