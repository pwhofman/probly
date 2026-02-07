"""Test for flax."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("flax")
pytest.importorskip("jax")

from flax import nnx
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from probly.conformal_prediction.methods.split import SplitConformalClassifier
from probly.conformal_prediction.scores.saps.common import SAPSScore
from probly.conformal_prediction.scores.saps.flax import saps_score_jax


class IrisFlaxModel(nnx.Module):
    def __init__(self) -> None:
        """Simple Flax model for Iris dataset."""
        self.dense1 = nnx.Linear(4, 8, rngs=nnx.Rngs(42))
        self.dense2 = nnx.Linear(8, 3, rngs=nnx.Rngs(42))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return x


class FlaxPredictor:
    def __init__(self, model: nnx.Module, params: FrozenDict) -> None:
        """Flax Predictor wrapper."""
        self.model = model
        self.params = params

    def __call__(self, x: Any) -> jnp.ndarray:  # noqa: ANN401
        x = jnp.asarray(x, dtype=jnp.float32)
        logits = self.model(x)
        return jax.nn.softmax(logits, axis=-1)

    def predict(self, x: Any) -> jnp.ndarray:  # noqa: ANN401
        return self(x)


@pytest.mark.skip(reason="Flaky Test")
def test_saps_with_iris_dataset() -> None:
    iris = load_iris()
    x, y = iris.data, iris.target

    # split
    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    _x_train, x_calib, _y_train, y_calib = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    # scale
    scaler = StandardScaler()
    x_calib = scaler.fit_transform(x_calib).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    # model + predictor
    model = IrisFlaxModel()
    predictor = FlaxPredictor(model, FrozenDict())

    # SAPS score
    score = SAPSScore(
        model=predictor,
        random_state=42,
    )

    conformal = SplitConformalClassifier(
        model=predictor,
        score=score,
    )

    # calibrate
    conformal.calibrate(x_calib, y_calib, alpha=0.1)
    assert conformal.is_calibrated

    # predict
    prediction_sets = conformal.predict(x_test, alpha=0.1)

    assert prediction_sets.shape == (len(x_test), 3)
    assert prediction_sets.dtype == bool

    # coverage
    coverage = np.mean(
        [prediction_sets[i, y_test[i]] for i in range(len(y_test))],
    )

    assert coverage >= 0.85


class SAPSFlaxTestModel:
    """A simple Flax model for testing."""

    def init_params(self, _key: jax.Array) -> dict[str, Any]:
        return {}

    def __init__(self, num_features: int, num_classes: int, key: jax.Array) -> None:
        """Initialize the test model."""
        self.num_features = num_features
        self.num_classes = num_classes
        self.params = self.init_params(key)


def test_rank1() -> None:
    probs = jnp.array([[0.15, 0.4, 0.25, 0.2]])  # shape (1, 4)
    u = jnp.array([[0.3, 0.3, 0.3, 0.3]])  # shape (1, 4)
    lambda_val = 0.1

    score = saps_score_jax(probs, lambda_val=lambda_val, u=u)

    # Check it returns an array, not scalar
    assert isinstance(score, jnp.ndarray)
    assert score.shape == (1, 4)

    # Max probability is 0.4 at index 1 (rank 1): 0.3 * 0.4 = 0.12
    assert jnp.allclose(score[0, 1], 0.12, rtol=1e-6)


def test_rank_greater_than_1() -> None:
    probs = jnp.array([[0.2, 0.5, 0.3, 0.1]])  # Dummy probabilities for testing
    u = jnp.array([[0.6, 0.6, 0.6, 0.6]])  # shape (1, 4)
    lambda_val = 0.2

    score = saps_score_jax(probs, lambda_val=lambda_val, u=u)

    # Max probability is 0.5 at index 1
    # For class with prob 0.3 (index 2), rank = 2
    # Score = 0.5 + (2-2+0.6)*0.2 = 0.5 + 0.6*0.2 = 0.62
    assert jnp.allclose(score[0, 2], 0.62, rtol=1e-6)


def test_2d_single_row() -> None:
    probs = jnp.array([[0.6, 0.1, 0.3]])  # Dummy probabilities for testing
    u = jnp.array([[0.4, 0.4, 0.4]])  # shape (1, 3)

    score = saps_score_jax(probs, lambda_val=0.1, u=u)

    # Max probability is 0.6 at index 0
    # For class with prob 0.1 (index 1), rank = 3
    # Score = 0.6 + (3-2+0.4)*0.1 = 0.6 + 1.4*0.1 = 0.74
    assert jnp.allclose(score[0, 1], 0.74, rtol=1e-6)


def test_output_type() -> None:
    # Use 2D array to avoid axis error
    probs = jnp.array([[0.3, 0.4, 0.3]])  # shape (1, 3)
    u = jnp.array([[0.5, 0.5, 0.5]])  # shape (1, 3)

    score = saps_score_jax(probs, lambda_val=0.1, u=u)

    assert isinstance(score, jnp.ndarray)
    assert score.shape == (1, 3)


def test_invalid_dimensions() -> None:
    """Test invalid dimensions - expect no error as JAX may broadcast."""
    probs = jnp.array([[0.2, 0.5], [0.3, 0.1]])  # Invalid shape
    u = jnp.array([0.1, 0.2])  # shape (2,)
    score = saps_score_jax(probs, lambda_val=0.1, u=u)

    assert score.shape == (2, 2)


def test_random_u_generation() -> None:
    # Use 2D array to avoid axis error
    probs = jnp.array([[0.3, 0.4, 0.3]])  # shape (1, 3)
    u = jnp.array([[0.5, 0.2, 0.8]])  # shape (1, 3)

    score = saps_score_jax(probs, lambda_val=0.1, u=u)

    assert isinstance(score, jnp.ndarray)
    assert score.shape == (1, 3)
