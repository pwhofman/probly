"""Tests for Flax APS implementation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from probly.conformal_prediction.methods.split import SplitConformal, SplitConformalClassifier
from probly.conformal_prediction.scores.aps.common import APSScore

pytest.importorskip("flax")
pytest.importorskip("jax")

from flax import nnx
from flax.core import FrozenDict
import jax
from jax import Array
import jax.numpy as jnp


class SimpleFlaxModel(nnx.Module):
    """Simple Flax model for testing."""

    def __init__(self) -> None:
        """Initialize model layers."""
        self.dense1 = nnx.Linear(10, 10, rngs=nnx.Rngs(0))
        self.dense2 = nnx.Linear(10, 3, rngs=nnx.Rngs(0))

    def __call__(self, x: Array) -> Array:
        """Forward pass."""
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)  # 3 output classes
        return x


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

        output = self.model(x_array)  # type: ignore[operator]
        logits = output[0] if isinstance(output, tuple) else output
        return jax.nn.softmax(logits, axis=-1)

    def predict(self, x: Any) -> Array:  # noqa: ANN401
        """Alias for __call__."""
        return self.__call__(x)


def create_test_data(
    n_samples: int = 100,
    n_features: int = 10,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Create test data for conformal prediction."""
    rng = np.random.RandomState(seed)

    # features as float32
    x_data = rng.randn(n_samples, n_features).astype(np.float32)

    # labels as int32
    y_data = rng.randint(0, n_classes, size=n_samples).astype(np.int32)

    return x_data, y_data


class TestAPSScoreFlax:
    """Tests for APSScore with Flax models."""

    @pytest.fixture
    def flax_model_and_params(self) -> tuple[nnx.Module, FrozenDict[str, Any]]:
        """Create a Flax model with initialized parameters."""
        model = SimpleFlaxModel()
        return model, cast(FrozenDict[str, Any], {})

    @pytest.fixture
    def flax_predictor(self, flax_model_and_params: tuple[nnx.Module, FrozenDict[str, Any]]) -> FlaxPredictor:
        """Create a Flax predictor for testing."""
        model, params = flax_model_and_params
        return FlaxPredictor(model, params)

    @pytest.fixture
    def test_data(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """Create test data."""
        return create_test_data(n_samples=50, n_features=10, n_classes=3)

    def test_flax_predictor_forward_pass_shapes(self, flax_predictor: FlaxPredictor) -> None:
        """Test that FlaxPredictor forward pass returns correct shapes."""
        rng = np.random.default_rng(42)
        x_test = rng.standard_normal((5, 10)).astype(np.float32)

        probs = flax_predictor(x_test)

        # check shapes
        assert probs.shape == (5, 3), f"Expected shape (5, 3), got {probs.shape}"

        # check probabilities sum to ~1
        np.testing.assert_allclose(
            jnp.sum(probs, axis=1),
            np.ones(5),
            rtol=1e-5,
        )

    def test_flax_predictor_output_types(self, flax_predictor: FlaxPredictor) -> None:
        """Test that FlaxPredictor outputs have correct dtypes."""
        rng = np.random.default_rng(42)
        x_test = rng.standard_normal((3, 10)).astype(np.float32)

        probs = flax_predictor(x_test)

        assert isinstance(probs, jnp.ndarray), f"Expected jnp.ndarray, got {type(probs)}"
        assert probs.dtype == jnp.float32, f"Expected float32, got {probs.dtype}"

    def test_flax_predictor_edge_case_shapes(self, flax_predictor: FlaxPredictor) -> None:
        """Test FlaxPredictor edge cases for input shapes."""
        rng = np.random.default_rng(42)

        # Test single sample
        x_single = rng.standard_normal((10,)).astype(np.float32)
        probs_single = flax_predictor(x_single)
        assert probs_single.shape == (3,), f"Expected shape (3,), got {probs_single.shape}"

        # Test large batch
        x_large = rng.standard_normal((100, 10)).astype(np.float32)
        probs_large = flax_predictor(x_large)
        assert probs_large.shape == (100, 3), f"Expected shape (100, 3), got {probs_large.shape}"

    def test_apsscore_with_flax_model(self, flax_predictor: FlaxPredictor) -> None:
        """Test APSScore with Flax model."""
        score = APSScore(model=flax_predictor, randomize=False)

        # create test data
        rng = np.random.default_rng(42)
        x_calib = rng.random((30, 10), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=30)

        # Test calibration scores
        cal_scores = score.calibration_nonconformity(x_calib, y_calib)

        cal_scores_np = np.asarray(cal_scores)
        assert cal_scores_np.shape == (30,)

        # Test prediction scores
        x_test = rng.random((10, 10), dtype=np.float32)
        pred_scores = score.predict_nonconformity(x_test)

        pred_scores_np = np.asarray(pred_scores)
        assert pred_scores_np.shape == (10, 3)

    def test_apsscore_integration(self, flax_predictor: FlaxPredictor) -> None:
        """Test APSScore integrated in SplitConformalClassifier."""
        score = APSScore(model=flax_predictor, randomize=False)
        cp_predictor = SplitConformalClassifier(model=flax_predictor, score=score)

        # create test data
        rng = np.random.default_rng(42)
        x_cal = rng.random((50, 10), dtype=np.float32)
        y_cal = rng.integers(0, 3, size=50)

        # calibrate
        threshold = cp_predictor.calibrate(x_cal, y_cal, alpha=0.1)

        assert cp_predictor.is_calibrated
        assert 0 <= threshold <= 1 + 1e-6  # Allow tolerance for float32 precision

        # predict
        x_test = rng.random((10, 10), dtype=np.float32)
        prediction_sets = cp_predictor.predict(x_test, alpha=0.1)

        prediction_sets_np = np.asarray(prediction_sets)
        assert prediction_sets_np.shape == (10, 3)
        assert prediction_sets_np.dtype in (bool, np.bool_)

    def test_with_different_random_states(self, flax_predictor: FlaxPredictor) -> None:
        """Test reproducibility with different random states."""
        # create two scores with same random state
        score1 = APSScore(model=flax_predictor, randomize=True)
        score2 = APSScore(model=flax_predictor, randomize=True)

        rng = np.random.default_rng(42)
        x_data = rng.random((20, 10), dtype=np.float32)
        y_data = rng.integers(0, 3, size=20)

        # should give same results with same random state
        scores1 = score1.calibration_nonconformity(x_data, y_data)
        scores2 = score2.calibration_nonconformity(x_data, y_data)

        assert np.allclose(scores1, scores2)

        # different random state should give different results
        score3 = APSScore(model=flax_predictor, randomize=True)
        scores3 = score3.calibration_nonconformity(x_data, y_data)

        # with randomization, they should be different
        assert not np.allclose(scores1, scores3)

    def test_with_and_without_randomization(self, flax_predictor: FlaxPredictor) -> None:
        """Compare scores with and without randomization."""
        score_no_rand = APSScore(model=flax_predictor, randomize=False)
        score_with_rand = APSScore(model=flax_predictor, randomize=True)

        rng = np.random.default_rng(42)
        x_data = rng.random((10, 10), dtype=np.float32)
        y_data = rng.integers(0, 3, size=10)

        scores_no_rand = score_no_rand.calibration_nonconformity(x_data, y_data)
        scores_with_rand = score_with_rand.calibration_nonconformity(x_data, y_data)

        # with randomization enabled, scores should be different
        assert not np.array_equal(scores_no_rand, scores_with_rand)

        # both should be in valid range (allow tolerance for float32 precision)
        assert bool(np.all(scores_no_rand <= 1 + 1e-6))
        assert bool(np.all(scores_no_rand >= 0))
        assert bool(np.all(scores_with_rand <= 1 + 1e-6))
        assert bool(np.all(scores_with_rand >= 0))

    def test_with_split_conformal(self, flax_predictor: FlaxPredictor) -> None:
        """Test integration with split conformal."""
        score = APSScore(model=flax_predictor, randomize=False)
        predictor = SplitConformalClassifier(model=flax_predictor, score=score)

        # create full dataset
        rng = np.random.default_rng(42)
        x_full = rng.random((150, 10), dtype=np.float32)
        y_full = rng.integers(0, 3, size=150)

        # create splitter
        splitter = SplitConformal(calibration_ratio=0.3)

        # split manually
        x_train, y_train, x_cal, y_cal = splitter.split(x_full, y_full)

        # calibrate
        predictor.calibrate(x_cal, y_cal, alpha=0.1)

        # verify calibration
        assert predictor.is_calibrated
        assert predictor.threshold is not None

        # make predictions
        x_test = rng.random((10, 10), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, alpha=0.1)

        assert prediction_sets.shape == (10, 3)

    def test_with_iris_dataset(self) -> None:
        """Test with real Iris dataset."""
        # load data
        iris = load_iris()
        x, y = iris.data, iris.target

        # split data
        x_temp, x_test, y_temp, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
        )

        x_train, x_calib, y_train, y_calib = train_test_split(
            x_temp,
            y_temp,
            test_size=0.25,
            stratify=y_temp,
        )

        # scale features
        scaler = StandardScaler()
        x_calib_scaled = scaler.fit_transform(x_calib).astype(np.float32)
        x_test_scaled = scaler.transform(x_test).astype(np.float32)

        # create Flax model for Iris (4 features, 3 classes)
        class IrisFlaxModel(nnx.Module):
            """Simple Flax model for Iris dataset."""

            def __init__(self) -> None:
                """Initialize model layers."""
                self.dense1 = nnx.Linear(4, 8, rngs=nnx.Rngs(42))
                self.dense2 = nnx.Linear(8, 3, rngs=nnx.Rngs(42))

            def __call__(self, x: Array) -> Array:
                """Forward pass."""
                x = self.dense1(x)
                x = jax.nn.relu(x)
                x = self.dense2(x)
                return x

        model = IrisFlaxModel()
        predictor_wrapper = FlaxPredictor(model, cast(FrozenDict[str, Any], {}))

        # create score and predictor
        score = APSScore(model=predictor_wrapper, randomize=False)
        predictor = SplitConformalClassifier(model=predictor_wrapper, score=score)

        # calibrate
        threshold = predictor.calibrate(x_calib_scaled, y_calib, alpha=0.1)

        assert predictor.is_calibrated
        assert 0 <= threshold <= 1 + 1e-6  # Allow tolerance for float32 precision

        # predict
        prediction_sets = predictor.predict(x_test_scaled, alpha=0.1)

        assert prediction_sets.shape == (len(x_test), 3)

        # calculate coverage
        covered = 0
        for i, true_label in enumerate(y_test):
            if prediction_sets[i, true_label]:
                covered += 1

        coverage = covered / len(y_test)

        # coverage should be reasonable (allow some flexibility since model isn't trained)
        assert 0.7 <= coverage <= 1.0

    def test_iris_coverage_guarantee(self) -> None:
        """Test that coverage guarantee holds on Iris dataset with multiple seeds."""
        # load data
        iris = load_iris()
        x, y = iris.data, iris.target

        # Test multiple random splits for robustness
        for seed in [42, 123, 456]:
            # Split data
            x_temp, x_test, y_temp, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                stratify=y,
            )

            x_train, x_calib, y_train, y_calib = train_test_split(
                x_temp,
                y_temp,
                test_size=0.25,
                stratify=y_temp,
            )

            # scale features
            scaler = StandardScaler()
            x_calib_scaled = scaler.fit_transform(x_calib).astype(np.float32)
            x_test_scaled = scaler.transform(x_test).astype(np.float32)

            # create Flax model with specific seed
            class SeededFlaxModel(nnx.Module):
                """Flax model with seeded initialization."""

                def __init__(self, model_seed: int) -> None:
                    """Initialize model layers with seed."""
                    self.dense1 = nnx.Linear(4, 8, rngs=nnx.Rngs(model_seed))
                    self.dense2 = nnx.Linear(8, 3, rngs=nnx.Rngs(model_seed))

                def __call__(self, x: Array) -> Array:
                    """Forward pass."""
                    x = self.dense1(x)
                    x = jax.nn.relu(x)
                    x = self.dense2(x)
                    return x

            model = SeededFlaxModel(model_seed=seed)
            predictor_wrapper = FlaxPredictor(model, cast(FrozenDict[str, Any], {}))

            # create and calibrate predictor
            score = APSScore(model=predictor_wrapper, randomize=False)
            predictor = SplitConformalClassifier(model=predictor_wrapper, score=score)

            threshold = predictor.calibrate(x_calib_scaled, y_calib, alpha=0.1)

            assert predictor.is_calibrated
            assert 0 <= threshold <= 1 + 1e-6  # Allow tolerance for float32 precision

            # predict
            prediction_sets = predictor.predict(x_test_scaled, alpha=0.1)

            assert prediction_sets.shape == (len(x_test), 3)

            # calculate coverage
            covered = 0
            for i, true_label in enumerate(y_test):
                if prediction_sets[i, true_label]:
                    covered += 1

            coverage = covered / len(y_test)

            # coverage should be >= 0.9 (1 - alpha), with some tolerance
            assert coverage >= 0.85, f"Coverage too low with seed {seed}: {coverage:.3f}"
