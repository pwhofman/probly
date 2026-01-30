"""Tests for Flax LAC implementation."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from probly.conformal_prediction.methods.split import SplitConformalClassifier
from probly.conformal_prediction.scores.lac.common import LACScore

pytest.importorskip("flax")
pytest.importorskip("jax")

from flax import nnx
from flax.core import FrozenDict
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import optax


class SimpleFlaxModel(nnx.Module):
    """Simple Flax model for testing."""

    def __init__(self) -> None:
        """Initialize model layers."""
        self.dense1 = nnx.Linear(4, 16, rngs=nnx.Rngs(0))
        self.dense2 = nnx.Linear(16, 3, rngs=nnx.Rngs(0))

    def __call__(self, x: Array) -> Array:
        """Forward pass."""
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return jax.nn.softmax(x)  # return probabilities directly


class FlaxPredictor:
    """Wrapper for Flax models to match Predictor protocol."""

    def __init__(self, model: nnx.Module, params: FrozenDict[str, Any] | dict[str, Any]) -> None:
        """Initialize the predictor."""
        self.model = model
        self.params = params

    def __call__(self, x: Any) -> Array:  # noqa: ANN401
        """Call the model and return probabilities."""
        if isinstance(x, np.ndarray):
            x_array = jnp.array(x, dtype=jnp.float32)
        else:
            x_array = jnp.array(np.asarray(x), dtype=jnp.float32)

        output = self.model(x_array)
        logits = output[0] if isinstance(output, tuple) else output
        return (Array, logits)

    def predict(self, x: Any) -> Array:  # noqa: ANN401
        """Alias for __call__."""
        return self.__call__(x)


@pytest.fixture
def flax_model_and_params() -> tuple[nnx.Module, FrozenDict[str, Any]]:
    """Create a Flax model with initialized parameters."""
    model = SimpleFlaxModel()
    return model, cast(FrozenDict[str, Any], {})


@pytest.fixture
def flax_predictor(flax_model_and_params: tuple[nnx.Module, FrozenDict[str, Any]]) -> FlaxPredictor:
    """Create a Flax predictor for testing."""
    model, params = flax_model_and_params
    return FlaxPredictor(model, params)


def test_lacscore_with_flax_model(flax_predictor: FlaxPredictor) -> None:
    """Test LACScore with Flax model."""
    score = LACScore(model=flax_predictor)

    # create test data
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


def test_lacscore_edge_case_single_sample(flax_predictor: FlaxPredictor) -> None:
    """Test LACScore with single sample."""
    score = LACScore(model=flax_predictor)
    x_calib = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y_calib = np.array([0])

    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert cal_scores.shape == (1,), f"expected shape (1,), got {cal_scores.shape}"
    assert np.all(cal_scores >= 0), "scores should be non-negative"
    assert np.all(cal_scores <= 1), "scores should be at most 1"


def test_lacscore_edge_case_large_batch(flax_predictor: FlaxPredictor) -> None:
    """Test LACScore with large batch."""
    score = LACScore(model=flax_predictor)
    rng = np.random.default_rng(42)
    x_calib = rng.random((500, 4), dtype=np.float32)
    y_calib = rng.integers(0, 3, size=500)

    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert cal_scores.shape == (500,), f"expected shape (500,), got {cal_scores.shape}"
    assert np.all(cal_scores >= 0), "scores should be non-negative"
    assert np.all(cal_scores <= 1), "scores should be at most 1"


def test_lacscore_output_types(flax_predictor: FlaxPredictor) -> None:
    """Test LACScore output types and dtypes."""
    score = LACScore(model=flax_predictor)
    rng = np.random.default_rng(42)
    x_test = rng.random((10, 4), dtype=np.float32)

    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray), f"expected np.ndarray, got {type(pred_scores)}"
    assert pred_scores.dtype in [np.float32, np.float64], f"expected float dtype, got {pred_scores.dtype}"
    assert pred_scores.shape == (10, 3), f"expected shape (10, 3), got {pred_scores.shape}"


def test_lacscore_multiple_classes(flax_predictor: FlaxPredictor) -> None:
    """Test LACScore with different number of classes."""
    score = LACScore(model=flax_predictor)
    rng = np.random.default_rng(42)

    # Test with different label values (simulating different classes)
    x_calib = rng.random((20, 4), dtype=np.float32)
    y_calib = rng.integers(0, 3, size=20)

    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert cal_scores.shape == (20,), f"expected shape (20,), got {cal_scores.shape}"
    assert np.all(cal_scores >= 0), "scores should be non-negative"
    assert np.all(cal_scores <= 1), "scores should be at most 1"


@pytest.mark.skipif(
    not hasattr(jax, "__version__"),
    reason="JAX not installed",
)
def test_lacscore_with_trained_flax_model() -> None:
    """Test LACScore with a trained Flax model on real Iris data."""
    # load and prepare iris data
    iris = load_iris()
    x_data, y_data = iris.data, iris.target

    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # split into train/temp
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        train_size=0.5,
        random_state=42,
    )

    # further split temp into calib/test
    x_calib, x_test, y_calib, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    # create and train model
    model = SimpleFlaxModel()
    tx = optax.adam(learning_rate=0.01)

    # separate parameters from other state
    params, state = nnx.split(model, nnx.Param)
    opt_state = tx.init(params)

    # training loop using nnx-style training
    x_train_jax = jnp.array(x_train, dtype=jnp.float32)
    y_train_jax = jnp.array(y_train, dtype=jnp.int32)

    @jax.jit
    def loss_fn(params: nnx.State, state: nnx.State, x: Array, y: Array) -> Array:
        """Compute cross-entropy loss."""
        model_test: SimpleFlaxModel = nnx.merge(params, state)  # type: ignore[arg-type]
        output = model_test(x)
        log_probs = jnp.log(output + 1e-8)
        return -jnp.mean(
            jnp.take_along_axis(
                log_probs,
                y[:, None],
                axis=1,
            ),
        )

    @jax.jit
    def train_step(
        params: nnx.State,
        state: nnx.State,
        opt_state: Any,  # noqa: ANN401
        x: Array,
        y: Array,
    ) -> tuple[nnx.State, Any, Array]:
        """Single training step with gradient updates."""
        loss, grads = jax.value_and_grad(loss_fn)(params, state, x, y)
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u if u is not None else p,
            params,
            updates,
        )
        return new_params, new_opt_state, loss

    # simple training loop
    for _ in range(50):
        params, opt_state, loss = train_step(params, state, opt_state, x_train_jax, y_train_jax)

    # merge back into model
    model = nnx.merge(params, state)

    # create predictor with trained model
    predictor = FlaxPredictor(model, {})
    score = LACScore(model=predictor)

    # Test calibration
    cal_scores = score.calibration_nonconformity(x_calib.astype(np.float32), y_calib)

    assert cal_scores.shape == (len(x_calib),)
    assert np.all(cal_scores >= 0)
    assert np.all(cal_scores <= 1)

    # Test prediction
    pred_scores = score.predict_nonconformity(x_test.astype(np.float32))

    assert pred_scores.shape == (len(x_test), 3)

    # create conformal predictor
    cp_predictor = SplitConformalClassifier(
        model=predictor,
        score=score,
        use_accretive=True,
    )

    # calibrate
    threshold = cp_predictor.calibrate(x_calib.astype(np.float32), y_calib, alpha=0.1)

    assert cp_predictor.is_calibrated
    assert 0 <= threshold <= 1 + 1e-6  # Allow tolerance for float32 precision

    # predict
    prediction_sets = cp_predictor.predict(x_test.astype(np.float32), alpha=0.1)

    assert prediction_sets.shape == (len(x_test), 3)
    assert prediction_sets.dtype == bool

    # check for non-empty sets (accretive should prevent them)
    set_sizes = np.sum(prediction_sets, axis=1)
    assert np.all(set_sizes >= 1)


@pytest.mark.skipif(
    not hasattr(jax, "__version__"),
    reason="JAX not installed",
)
def test_lacscore_iris_coverage_guarantee() -> None:
    """Test LACScore with Iris data and verify coverage guarantee."""
    # load and prepare iris data
    iris = load_iris()
    x_data, y_data = iris.data, iris.target

    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # split into train/temp
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        train_size=0.5,
        random_state=42,
    )

    # further split temp into calib/test
    x_calib, x_test, y_calib, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    # create and train model
    model = SimpleFlaxModel()
    tx = optax.adam(learning_rate=0.01)

    # separate parameters from other state
    params, state = nnx.split(model, nnx.Param)
    opt_state = tx.init(params)

    # training loop using nnx-style training
    x_train_jax = jnp.array(x_train, dtype=jnp.float32)
    y_train_jax = jnp.array(y_train, dtype=jnp.int32)

    @jax.jit
    def loss_fn(params: nnx.State, state: nnx.State, x: Array, y: Array) -> Array:
        """Compute cross-entropy loss."""
        model_test: SimpleFlaxModel = nnx.merge(params, state)  # type: ignore[arg-type]
        output = model_test(x)
        log_probs = jnp.log(output + 1e-8)
        return -jnp.mean(
            jnp.take_along_axis(
                log_probs,
                y[:, None],
                axis=1,
            ),
        )

    @jax.jit
    def train_step(
        params: nnx.State,
        state: nnx.State,
        opt_state: Any,  # noqa: ANN401
        x: Array,
        y: Array,
    ) -> tuple[nnx.State, Any, Array]:
        """Single training step with gradient updates."""
        loss, grads = jax.value_and_grad(loss_fn)(params, state, x, y)
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u if u is not None else p,
            params,
            updates,
        )
        return new_params, new_opt_state, loss

    # simple training loop
    for _ in range(50):
        params, opt_state, loss = train_step(params, state, opt_state, x_train_jax, y_train_jax)

    # merge back into model
    model = nnx.merge(params, state)

    # create predictor with trained model
    predictor = FlaxPredictor(model, {})
    score = LACScore(model=predictor)

    # create conformal predictor
    cp_predictor = SplitConformalClassifier(
        model=predictor,
        score=score,
        use_accretive=True,
    )

    # calibrate with alpha=0.1 (90% coverage)
    cp_predictor.calibrate(x_calib.astype(np.float32), y_calib, alpha=0.1)

    # predict on test set
    prediction_sets = cp_predictor.predict(x_test.astype(np.float32), alpha=0.1)

    # check shape
    assert prediction_sets.shape == (len(x_test), 3)

    # compute empirical coverage: what fraction of test samples have true label in prediction set
    coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(len(y_test))])

    # coverage should be at least 1 - alpha = 0.9
    assert coverage >= 0.85, f"coverage {coverage} is below expected 0.85"

    # check that all sets are non-empty (accretive completion)
    set_sizes = np.sum(prediction_sets, axis=1)
    assert np.all(set_sizes >= 1), "all prediction sets should be non-empty with accretive completion"


@pytest.mark.skipif(
    not hasattr(jax, "__version__"),
    reason="JAX not installed",
)
def test_lacscore_iris_multiple_seeds() -> None:
    """Test LACScore with Iris data across multiple random seeds."""
    iris = load_iris()
    x_data, y_data = iris.data, iris.target

    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # split into train/temp
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        train_size=0.5,
        random_state=42,
    )

    # further split temp into calib/test
    x_calib, x_test, y_calib, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    coverages = []

    # Test with multiple random seeds
    for seed in [42, 123, 456]:
        # create and train model with new seed
        model = SimpleFlaxModel()
        tx = optax.adam(learning_rate=0.01)

        # separate parameters from other state
        params, state = nnx.split(model, nnx.Param)
        opt_state = tx.init(params)

        # training loop using nnx-style training
        x_train_jax = jnp.array(x_train, dtype=jnp.float32)
        y_train_jax = jnp.array(y_train, dtype=jnp.int32)

        @jax.jit
        def loss_fn(params: nnx.State, state: nnx.State, x: Array, y: Array) -> Array:
            """Compute cross-entropy loss."""
            model_test: SimpleFlaxModel = nnx.merge(params, state)  # type: ignore[arg-type]
            output = model_test(x)
            log_probs = jnp.log(output + 1e-8)
            return -jnp.mean(
                jnp.take_along_axis(
                    log_probs,
                    y[:, None],
                    axis=1,
                ),
            )

        def train_step(
            tx_local: Any,  # noqa: ANN401
            params: nnx.State,
            state: nnx.State,
            opt_state: Any,  # noqa: ANN401
            x: Array,
            y: Array,
        ) -> tuple[nnx.State, Any, Array]:
            """Single training step with gradient updates."""
            loss, grads = jax.value_and_grad(loss_fn)(params, state, x, y)
            updates, new_opt_state = tx_local.update(grads, opt_state)
            new_params = jax.tree_util.tree_map(
                lambda p, u: p + u if u is not None else p,
                params,
                updates,
            )
            return new_params, new_opt_state, loss

        # simple training loop
        for _ in range(50):
            params, opt_state, loss = train_step(tx, params, state, opt_state, x_train_jax, y_train_jax)  # type: ignore[assignment, arg-type]

        # merge back into model
        model = nnx.merge(params, state)

        # create predictor with trained model
        predictor = FlaxPredictor(model, {})
        score = LACScore(model=predictor)

        # create conformal predictor
        cp_predictor = SplitConformalClassifier(
            model=predictor,
            score=score,
            use_accretive=True,
        )

        # calibrate with alpha=0.1
        cp_predictor.calibrate(x_calib.astype(np.float32), y_calib, alpha=0.1)

        # predict on test set
        prediction_sets = cp_predictor.predict(x_test.astype(np.float32), alpha=0.1)

        # compute empirical coverage
        coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(len(y_test))])
        coverages.append(coverage)

        # coverage should be at least 0.85
        assert coverage >= 0.85, f"coverage {coverage} is below expected 0.85 for seed {seed}"

    # verify coverage is non-empty across all seeds
    assert len(coverages) == 3
    assert all(c >= 0.85 for c in coverages)
