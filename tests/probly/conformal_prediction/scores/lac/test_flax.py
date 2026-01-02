"""Tests for Flax LAC implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("flax")
pytest.importorskip("jax")

if TYPE_CHECKING:
    pass

from flax.core import FrozenDict
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from probly.conformal_prediction.methods.split import SplitConformalPredictor
from probly.conformal_prediction.scores.lac.common import LACScore


class SimpleFlaxModel(nn.Module):
    """Simple Flax model for testing."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass."""
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # 3 output classes for Iris
        return nn.softmax(x)  # Return probabilities directly


class FlaxPredictor:
    """Wrapper for Flax models to match Predictor protocol."""

    def __init__(self, model: nn.Module, params: FrozenDict[str, Any] | dict[str, Any]) -> None:
        """Initialize the predictor."""
        self.model = model
        self.params = params

    def __call__(self, x: Any) -> jax.Array:  # noqa: ANN401
        """Call the model and return probabilities."""
        if isinstance(x, np.ndarray):
            x_array = jnp.array(x, dtype=jnp.float32)
        else:
            x_array = jnp.array(np.asarray(x), dtype=jnp.float32)

        output = self.model.apply(self.params, x_array)
        logits = output[0] if isinstance(output, tuple) else output
        return cast(jax.Array, logits)

    def predict(self, x: Any) -> jax.Array:  # noqa: ANN401
        """Alias for __call__."""
        return self.__call__(x)


@pytest.fixture
def flax_model_and_params() -> tuple[nn.Module, FrozenDict[str, Any]]:
    """Create a Flax model with initialized parameters."""
    model = SimpleFlaxModel()
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 4))
    params = model.init(key, dummy_input)

    return model, cast(FrozenDict[str, Any], params)


@pytest.fixture
def flax_predictor(flax_model_and_params: tuple[nn.Module, FrozenDict[str, Any]]) -> FlaxPredictor:
    """Create a Flax predictor for testing."""
    model, params = flax_model_and_params
    return FlaxPredictor(model, params)


def test_lacscore_with_flax_model(flax_predictor: FlaxPredictor) -> None:
    """Test LACScore with Flax model."""
    score = LACScore(model=flax_predictor)

    # Create test data
    rng = np.random.default_rng(42)
    x_calib = rng.random((30, 4), dtype=np.float32)
    y_calib = rng.integers(0, 3, size=30)

    # Test calibration scores
    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (30,)
    assert np.all(cal_scores >= 0)
    assert np.all(cal_scores <= 1)

    # Test prediction scores
    x_test = rng.random((10, 4), dtype=np.float32)
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (10, 3)


@pytest.mark.skipif(
    not hasattr(jax, "__version__"),
    reason="JAX not installed",
)
def test_lacscore_with_trained_flax_model(flax_model_and_params: tuple[nn.Module, FrozenDict[str, Any]]) -> None:
    """Test LACScore with a trained Flax model on Iris."""
    # Load and prepare data
    iris = load_iris()
    x_data, y_data = iris.data, iris.target

    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # Split data
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

    model, params = flax_model_and_params

    # Training setup
    tx = optax.adam(learning_rate=0.01)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(
        params: FrozenDict[str, Any] | dict[str, Any],
        opt_state: Any,  # noqa: ANN401
        batch_x: jax.Array,
        batch_y: jax.Array,
    ) -> tuple[FrozenDict[str, Any] | dict[str, Any], Any, jax.Array]:
        def loss_fn(params: FrozenDict[str, Any] | dict[str, Any]) -> jax.Array:
            output = model.apply(params, batch_x)
            # Extract logits if apply returns a tuple
            logits = output[0] if isinstance(output, tuple) else output
            # Cross-entropy loss
            log_probs = jnp.log(logits + 1e-8)
            return -jnp.mean(
                jnp.take_along_axis(
                    log_probs,
                    batch_y[:, None],
                    axis=1,
                ),
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Convert to JAX arrays
    x_train_jax = jnp.array(x_train, dtype=jnp.float32)
    y_train_jax = jnp.array(y_train, dtype=jnp.int32)

    # Simple training
    for _ in range(50):
        params, opt_state, loss = train_step(
            params,
            opt_state,
            x_train_jax,
            y_train_jax,
        )

    # Create predictor with trained params
    predictor_obj = FlaxPredictor(model, params)
    score = LACScore(model=predictor_obj)

    # Test calibration
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert cal_scores.shape == (len(x_cal),)
    assert np.all(cal_scores >= 0)
    assert np.all(cal_scores <= 1)

    # Test prediction
    pred_scores = score.predict_nonconformity(x_test)

    assert pred_scores.shape == (len(x_test), 3)

    # Create conformal predictor
    cp_predictor = SplitConformalPredictor(
        model=predictor_obj,
        score=score,
        use_accretive=True,
    )

    # Calibrate
    threshold = cp_predictor.calibrate(x_cal, y_cal, alpha=0.1)

    assert cp_predictor.is_calibrated
    assert 0 <= threshold <= 1

    # Predict
    prediction_sets = cp_predictor.predict(x_test, alpha=0.1)

    assert prediction_sets.shape == (len(x_test), 3)
    assert prediction_sets.dtype == bool

    # Check for empty sets (accretive should prevent them)
    set_sizes = np.sum(prediction_sets, axis=1)
    assert np.all(set_sizes >= 1)
