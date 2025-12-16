"""Tests for Flax APS implementation."""

from __future__ import annotations

import pytest

pytest.importorskip("flax")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from probly.conformal_prediction.aps.flax import FlaxAPS


def create_test_data(
    n_samples: int = 100,
    n_features: int = 20,
    n_classes: int = 10,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Create test data for conformal prediction."""
    rng = np.random.RandomState(seed)

    # Features as float32
    x_data = rng.randn(n_samples, n_features).astype(np.float32)

    # Labels as int32
    y_data = rng.randint(0, n_classes, size=n_samples).astype(np.int32)

    return x_data, y_data


def calculate_coverage(prediction_sets: list[set[int]], true_labels: npt.NDArray[np.int32]) -> float:
    """Calculate coverage percentage."""
    correct = 0
    for pred_set, true_label in zip(prediction_sets, true_labels, strict=False):
        if true_label in pred_set:
            correct += 1
    return correct / len(true_labels)


class TestFlaxAPS:
    """Tests for FlaxAPS class."""

    @pytest.fixture
    def simple_nnx_model(self) -> nnx.Module:
        """Create a simple Flax model."""

        class SimpleNNXModel(nnx.Module):
            def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs) -> None:
                """Initialize model layers."""
                super().__init__()
                self.dense = nnx.Linear(din, dout, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                """Forward pass."""
                return self.dense(x)

        rngs = nnx.Rngs(42)
        model = SimpleNNXModel(din=10, dout=3, rngs=rngs)

        return model

    @pytest.fixture
    def test_data(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """Create test data."""
        return create_test_data(n_samples=50, n_features=10, n_classes=3)

    def test_initialization(self, simple_nnx_model: nnx.Module) -> None:
        """Test that FlaxAPS initializes correctly."""
        model = simple_nnx_model

        # Test different rng_key types
        predictor_int = FlaxAPS(model, rng_key=42)
        assert predictor_int.rng is not None

        predictor_key = FlaxAPS(model, rng_key=jax.random.PRNGKey(123))
        assert predictor_key.rng is not None

        predictor_none = FlaxAPS(model, rng_key=None)
        assert predictor_none.rng is not None

        # Check attributes
        assert predictor_int.flax_model is model

    def test_forward_pass_shapes(self, simple_nnx_model: nnx.Module) -> None:
        """Test that forward pass returns correct shapes."""
        model = simple_nnx_model
        predictor = FlaxAPS(model)

        rng = np.random.default_rng(42)

        # Test input shapes
        batch_size = 5
        n_features = 10
        n_classes = 3

        x_test = rng.standard_normal((batch_size, n_features)).astype(np.float32)

        # Get predictions
        probs = predictor.model.predict(x_test)

        # Check shapes
        assert probs.shape == (batch_size, n_classes), f"Expected shape {(batch_size, n_classes)}, got {probs.shape}"

        # Check probabilities sum to ~1
        np.testing.assert_allclose(
            probs.sum(axis=1),
            np.ones(batch_size),
            rtol=1e-5,
        )

    def test_output_types(self, simple_nnx_model: nnx.Module) -> None:
        """Test that outputs have correct dtypes."""
        model = simple_nnx_model
        predictor = FlaxAPS(model)

        rng = np.random.default_rng(42)

        x_test = rng.standard_normal((3, 10)).astype(np.float32)

        # Test jit_predict returns jax array
        probs_jit = predictor.jit_predict(jnp.array(x_test))
        assert isinstance(probs_jit, jnp.ndarray), f"Expected jnp.ndarray, got {type(probs_jit)}"
        assert probs_jit.dtype == jnp.float32, f"Expected float32, got {probs_jit.dtype}"

    def test_edge_case_shapes(self, simple_nnx_model: nnx.Module) -> None:
        """Test edge cases for input shapes."""
        model = simple_nnx_model
        predictor = FlaxAPS(model)

        rng = np.random.default_rng(42)

        # Test single sample
        x_single = rng.standard_normal((10,)).astype(np.float32)
        probs_single = predictor.model.predict([x_single])
        assert probs_single.shape == (1, 3)

        # Test large batch
        x_large = rng.standard_normal((1000, 10)).astype(np.float32)
        probs_large = predictor.model.predict(x_large)
        assert probs_large.shape == (1000, 3)

    def test_compute_nonconformity(
        self,
        simple_nnx_model: nnx.Module,
        test_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    ) -> None:
        """Test nonconformity score computation."""
        x_data, y_data = test_data
        model = simple_nnx_model

        predictor = FlaxAPS(model)

        # Test with a subset of data
        x_subset = x_data[:10]
        y_subset = y_data[:10]

        # Calibrate to compute scores
        predictor.calibrate(x_subset, y_subset, significance_level=0.1)

        # Check that scores were computed
        assert predictor.nonconformity_scores is not None
        scores = predictor.nonconformity_scores

        # Check shape and dtype
        assert scores.shape == (10,)
        assert scores.dtype == np.float32

        # APS scores should be in [0, 1]
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_calibration(
        self,
        simple_nnx_model: nnx.Module,
        test_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    ) -> None:
        """Test calibration process."""
        x_data, y_data = test_data
        model = simple_nnx_model

        predictor = FlaxAPS(model)

        # Split data for calibration
        split_idx = 30
        x_cal = x_data[:split_idx]
        y_cal = y_data[:split_idx]

        # Calibrate
        threshold = predictor.calibrate(x_cal, y_cal, significance_level=0.1)

        # Check results
        assert predictor.is_calibrated
        assert predictor.threshold is not None
        assert threshold is not None
        assert 0 <= threshold <= 1

        # Check that nonconformity scores are stored
        assert predictor.nonconformity_scores is not None
        assert len(predictor.nonconformity_scores) == split_idx

    def test_prediction_before_calibration_fails(
        self,
        simple_nnx_model: nnx.Module,
        test_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    ) -> None:
        """Test that predict() fails before calibration."""
        x_data, _ = test_data
        model = simple_nnx_model

        predictor = FlaxAPS(model)

        with pytest.raises(ValueError, match="Call calibrate"):
            predictor.predict(x_data[:5], 0.1)

    def test_prediction_after_calibration(
        self,
        simple_nnx_model: nnx.Module,
        test_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    ) -> None:
        """Test prediction after calibration with different scenarios."""
        model = simple_nnx_model
        x_data, y_data = test_data

        predictor = FlaxAPS(model)

        # Test normal prediction
        x_cal = x_data[:30]
        y_cal = y_data[:30]
        predictor.calibrate(x_cal, y_cal, significance_level=0.1)

        x_test = x_data[30:40]
        prediction_sets = predictor.predict(x_test, 0.1)

        assert len(prediction_sets) == 10
        for pred_set in prediction_sets:
            assert isinstance(pred_set, set)
            assert len(pred_set) >= 1
            assert all(0 <= idx < 3 for idx in pred_set)

        # Test with different random state
        rng = np.random.RandomState(123)
        x_dummy = rng.randn(3, 10).astype(np.float32)
        prediction_sets2 = predictor.predict(x_dummy, 0.1)

        assert len(prediction_sets2) == 3

    def test_jit_predict(
        self,
        simple_nnx_model: nnx.Module,
        test_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    ) -> None:
        """Test JIT-compiled prediction."""
        x_data, _ = test_data
        model = simple_nnx_model

        predictor = FlaxAPS(model)

        # Convert to JAX array
        x_jax = jnp.array(x_data[:5], dtype=jnp.float32)

        # Get probabilities via jit_predict
        probs_jit = predictor.jit_predict(x_jax)

        # Get probabilities via regular path
        x_np = np.array(x_data[:5])
        probs_regular = predictor.model.predict(x_np)

        # Check shape and similarity
        assert probs_jit.shape == (5, 3)
        np.testing.assert_allclose(np.array(probs_jit), probs_regular, rtol=1e-5, atol=1e-5)

    def test_str_representation(
        self,
        simple_nnx_model: nnx.Module,
    ) -> None:
        """Test string representation."""
        model = simple_nnx_model

        predictor = FlaxAPS(model)

        # Before calibration
        repr_str = str(predictor)
        assert "FlaxAPS" in repr_str
        assert "SimpleNNXModel" in repr_str
        assert "not calibrated" in repr_str

        # After calibration
        rng = np.random.RandomState(42)
        x_dummy = rng.randn(10, 10).astype(np.float32)
        y_dummy = rng.randint(0, 3, size=10).astype(np.int32)
        predictor.calibrate(x_dummy, y_dummy, significance_level=0.1)

        repr_str_calibrated = str(predictor)
        assert "calibrated" in repr_str_calibrated

    def test_fit_with_split(
        self,
        simple_nnx_model: nnx.Module,
        test_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    ) -> None:
        """Test that fit_with_split method works from parent class."""
        x_data, y_data = test_data
        model = simple_nnx_model

        predictor = FlaxAPS(model)

        # Use fit_with_split from parent class
        x_train, y_train = predictor.fit_with_split(
            x=x_data,
            y=y_data,
            significance_level=0.1,
            calibration_ratio=0.3,
        )

        # Check that predictor is calibrated
        assert predictor.is_calibrated
        assert predictor.threshold is not None

        # Check split info
        split_info = predictor.get_split_info()
        assert split_info is not None


class TestFlaxAPSwithIris:
    """Iris dataset tests for FlaxAPS."""

    @pytest.fixture
    def iris_data(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """Load Iris dataset with correct dtypes."""
        iris = load_iris()
        x = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        return x, y

    @pytest.fixture
    def iris_nnx_model(self) -> nnx.Module:
        """Create a simple Iris model."""

        class IrisModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                """Initialize irismodel with nnx."""
                super().__init__()
                self.dense1 = nnx.Linear(4, 10, rngs=rngs)
                self.dense2 = nnx.Linear(10, 3, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                x = jax.nn.relu(self.dense1(x))
                return self.dense2(x)

        rngs = nnx.Rngs(42)
        return IrisModel(rngs=rngs)

    def test_iris_dataset_basic(
        self,
        iris_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
        iris_nnx_model: nnx.Module,
    ) -> None:
        """Basic test with Iris dataset."""
        x, y = iris_data
        model = iris_nnx_model

        # Split data for calibration
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Further split for calibration
        x_train_final, x_cal, y_train_final, y_cal = train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=42,
            stratify=y_train,
        )

        # Create predictor
        predictor = FlaxAPS(model, rng_key=42)

        # Calibrate
        threshold = predictor.calibrate(x_cal, y_cal, significance_level=0.1)

        # Basic assertions
        assert predictor.is_calibrated
        assert threshold is not None
        assert 0 <= threshold <= 1

        # Predict on test set
        prediction_sets = predictor.predict(x_test, 0.1)

        # Check predictions
        assert len(prediction_sets) == len(x_test)
        for pred_set in prediction_sets:
            assert isinstance(pred_set, set)
            assert len(pred_set) >= 1  # At least one class
            assert all(0 <= idx < 3 for idx in pred_set)

        # Calculate coverage
        coverage = calculate_coverage(prediction_sets, y_test)

        # Coverage should be approximately 1 - alpha (0.9)
        assert coverage >= 0.85, f"Coverage too low: {coverage:.3f}"

    def test_iris_coverage_guarantee(
        self,
        iris_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]],
    ) -> None:
        """Test that coverage guarantee holds on Iris dataset."""
        x, y = iris_data

        # Test multiple random splits for robustness
        for seed in [42, 123, 456]:
            # Create model
            class SimpleModel(nnx.Module):
                def __init__(self, *, rngs: nnx.Rngs) -> None:
                    super().__init__()
                    self.dense1 = nnx.Linear(4, 8, rngs=rngs)
                    self.dense2 = nnx.Linear(8, 3, rngs=rngs)

                def __call__(self, x: jax.Array) -> jax.Array:
                    x = jax.nn.relu(self.dense1(x))
                    return self.dense2(x)

            rngs = nnx.Rngs(seed)
            model = SimpleModel(rngs=rngs)

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=y,
            )

            # Split calibration set
            x_train_final, x_cal, y_train_final, y_cal = train_test_split(
                x_train,
                y_train,
                test_size=0.25,
                random_state=seed,
                stratify=y_train,
            )

            # Create and calibrate predictor
            predictor = FlaxAPS(model, rng_key=seed)
            predictor.calibrate(x_cal, y_cal, significance_level=0.1)

            # Predict
            prediction_sets = predictor.predict(x_test, 0.1)

            # Calculate coverage
            coverage = calculate_coverage(prediction_sets, y_test)

            # Coverage should be >= 0.9 (1 - alpha)
            assert coverage >= 0.85, f"Coverage too low with seed {seed}: {coverage:.3f}"
