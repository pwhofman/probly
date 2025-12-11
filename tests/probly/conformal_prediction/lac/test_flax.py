"""Tests for the Flax implementation of LAC including Iris integration."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("flax")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from probly.conformal_prediction.lac.common import (
    calculate_weighted_quantile,
)
from probly.conformal_prediction.lac.flax import LACFlax


class MockFlaxModel:
    """Simulates a Flax module for unit testing."""

    def apply(self, params: Any, x: Any) -> Any:  # noqa: ANN401, ARG002
        """Return fixed logits."""
        n_samples = x.shape[0]
        logits = jnp.array([[1.0, 0.5, 0.2]])
        return jnp.tile(logits, (n_samples, 1))


def test_flax_prediction_flow() -> None:
    """Test that LACFlax runs through predict pipeline."""
    model = MockFlaxModel()
    params: dict[str, Any] = {}
    predictor = LACFlax(model, params)

    predictor.is_calibrated = True
    predictor.threshold = 0.8

    x_dummy = np.zeros((2, 5))
    sets = predictor.predict(x_dummy, significance_level=0.1)

    assert len(sets) == 2
    assert isinstance(sets[0], np.ndarray)


def test_flax_nonconformity() -> None:
    """Test calculation of scores using Flax wrapper."""
    model = MockFlaxModel()
    predictor = LACFlax(model, params={})

    x_dummy = np.zeros((3, 5))
    y_dummy = np.array([0, 1, 2])

    scores = predictor._compute_nonconformity(x_dummy, y_dummy)  # noqa: SLF001

    assert scores.shape == (3,)
    assert np.all(scores >= 0.0)


# REAL WORLD DATASET TEST (IRIS)


class IrisMLP(nn.Module):
    """A simple MLP for Iris classification in Flax."""

    @nn.compact
    def __call__(self, x: Any) -> Any:  # noqa: ANN401
        """Forward pass."""
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # 3 Classes for Iris
        return x


def test_iris_flax_integration() -> None:
    """Train a simple Flax model on Iris and test LAC wrapper."""
    # 1. Prepare Data
    iris = load_iris()
    # Use lowercase variable names to satisfy linter (N806)
    x_data, y_data = iris.data, iris.target
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # Split
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        train_size=0.5,
        random_state=42,
    )
    x_cal, x_test, y_cal, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    # 2. Initialize Flax Model
    model = IrisMLP()
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)

    # 3. Quick Training Loop (using Optax)
    tx = optax.adam(learning_rate=0.01)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(
        params: Any,  # noqa: ANN401
        opt_state: Any,  # noqa: ANN401
        batch_x: Any,  # noqa: ANN401
        batch_y: Any,  # noqa: ANN401
    ) -> tuple[Any, Any]:
        def loss_fn(params: Any) -> Any:  # noqa: ANN401
            logits = model.apply(params, batch_x)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits,
                    labels=batch_y,
                ),
            )

        grad_fn = jax.value_and_grad(loss_fn)
        _, grads = grad_fn(params)
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    x_train_jax = jnp.array(x_train)
    y_train_jax = jnp.array(y_train)

    for _ in range(50):
        params, opt_state = train_step(params, opt_state, x_train_jax, y_train_jax)

    # 4. Wrap in LACFlax
    predictor = LACFlax(model, params)

    # 5. Calibration
    cal_scores = predictor._compute_nonconformity(x_cal, y_cal)  # noqa: SLF001

    alpha = 0.1
    # - alpha was 1.0 - alpha, ensure correct subtraction
    q_val = calculate_weighted_quantile(cal_scores, quantile=1.0 - alpha)
    predictor.is_calibrated = True
    predictor.threshold = q_val

    # 6. Predict on Test Set
    prediction_sets = predictor.predict(x_test, significance_level=alpha)

    # 7. Metrics
    set_sizes = [np.sum(s) for s in prediction_sets]
    covered = [p_set[y_t] for p_set, y_t in zip(prediction_sets, y_test, strict=False)]
    coverage = np.mean(covered)
    empty_rate = np.mean([s == 0 for s in set_sizes])

    # 8. Assertions
    assert coverage >= 0.75, f"Flax coverage too low: {coverage:.2f}"
    assert empty_rate == 0.0, "Flax produced empty sets!"
