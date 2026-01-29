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
        logits = self.model(x)  # type: ignore[operator]
        return jax.nn.softmax(logits, axis=-1)

    def predict(self, x: Any) -> jnp.ndarray:  # noqa: ANN401
        return self(x)


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

    x_train, x_calib, y_train, y_calib = train_test_split(
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
    probs = jnp.array([0.15, 0.4, 0.25, 0.2])
    label = 2
    u = 0.3
    lambda_val = 0.1

    score = saps_score_jax(probs, label, lambda_val, u)

    sorted_probs = sorted(probs, reverse=True)
    rank = sorted_probs.index(probs[label])
    max_prob_in_set = max(probs)
    expected = float(max_prob_in_set + lambda_val * (rank + u))

    assert score == pytest.approx(expected)


def test_rank_greater_than_1() -> None:
    probs = jnp.array([0.2, 0.5, 0.3, 0.1])  # Dummy probabilities for testing
    label = 2
    u = 0.3
    lambda_val = 0.2

    score = saps_score_jax(probs, label, lambda_val, u)

    max_prob = 0.5
    expected = max_prob + lambda_val * (1 + u)
    assert score == expected


def test_2d_single_row() -> None:
    probs = jnp.array([[0.6, 0.1, 0.3]])  # Dummy probabilities for testing
    label = 1
    u = 0.4

    score = saps_score_jax(probs, label, lambda_val=0.1, u=u)

    max_prob = 0.6
    expected = max_prob + 0.1 * (1 + u)
    assert score == pytest.approx(expected)


def test_output_type() -> None:
    probs = jnp.array([0.3, 0.4, 0.3])  # Dummy probabilities for testing
    label = 0
    u = 0.1

    score = saps_score_jax(probs, label, lambda_val=0.1, u=u)

    assert isinstance(score, float)


def test_invalid_dimensions() -> None:
    probs = jnp.array([[0.2, 0.5], [0.3, 0.1]])  # Invalid shape
    label = 0

    with pytest.raises(ValueError):  # noqa: PT011
        saps_score_jax(probs, label, lambda_val=0.1)


@pytest.mark.skip(reason="JAX does not raise IndexError for out of bounds access by design")
def test_label_out_of_bounds() -> None:
    probs = jnp.array([0.2, 0.5, 0.3])  # Dummy probabilities for testing
    label = 3  # Invalid label

    with pytest.raises(ValueError):  # noqa: PT011
        saps_score_jax(probs, label, lambda_val=0.1)


def test_random_u_generation() -> None:
    probs = jnp.array([0.3, 0.4, 0.3])  # Dummy probabilities for testing
    label = 0

    score1 = saps_score_jax(probs, label, lambda_val=0.1)

    assert isinstance(score1, float)
