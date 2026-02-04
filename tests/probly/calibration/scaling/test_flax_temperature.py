"""Tests for the temperature scaling implementation with flax."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from probly.calibration.scaling.flax_temperature import FlaxTemperature


def test_forward(flax_setup_multiclass: nnx.Module) -> None:
    base, inputs, _ = flax_setup_multiclass
    temperature_model = FlaxTemperature(base, num_classes=3)

    logits_base = base(inputs)
    logits_expected = logits_base / temperature_model.temperature
    logits_scaled = temperature_model(inputs)

    assert jnp.allclose(logits_scaled, logits_expected, atol=1e-5)


def test_fit(flax_setup_multiclass: nnx.Module) -> None:
    base, inputs, labels = flax_setup_multiclass
    temperature_model = FlaxTemperature(base, num_classes=3)

    dataloader = [(inputs, labels)]
    temperature_unoptimized = temperature_model.temperature.clone()

    temperature_model.fit(dataloader, learning_rate=0.01, max_iter=50)

    assert not jnp.allclose(temperature_model.temperature, temperature_unoptimized)


def test_predict(flax_setup_multiclass: nnx.Module) -> None:
    base, inputs, _ = flax_setup_multiclass
    temperature_model = FlaxTemperature(base, num_classes=3)

    predictions = temperature_model.predict(inputs)
    row_sums = predictions.sum(axis=1)

    assert predictions.shape == (20, 3)
    assert jnp.all(predictions >= 0)
    assert jnp.all(predictions <= 1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums))
