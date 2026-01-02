"""Tests for Flax APS implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from flax.core import FrozenDict
import pytest

pytest.importorskip("flax")
pytest.importorskip("jax")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.methods.split import SplitConformalPredictor
from probly.conformal_prediction.scores.aps.common import APSScore

# TYPE_CHECKING nur für Sequence (wenn nötig)
if TYPE_CHECKING:
    pass


class SimpleFlaxModel(nn.Module):
    """Simple Flax model for testing."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass."""
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # 3 output classes
        return x


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
        return jax.nn.softmax(logits, axis=-1)

    def predict(self, x: Any) -> jax.Array:  # noqa: ANN401
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

    # Features as float32
    x_data = rng.randn(n_samples, n_features).astype(np.float32)

    # Labels as int32
    y_data = rng.randint(0, n_classes, size=n_samples).astype(np.int32)

    return x_data, y_data


class TestAPSScoreFlax:
    """Tests for APSScore with Flax models."""

    @pytest.fixture
    def flax_model_and_params(self) -> tuple[nn.Module, FrozenDict[str, Any]]:
        """Create a Flax model with initialized parameters."""
        model = SimpleFlaxModel()
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 10), dtype=jnp.float32)
        params = model.init(key, dummy_input)
        return model, cast(FrozenDict[str, Any], params)

    @pytest.fixture
    def flax_predictor(self, flax_model_and_params: tuple[nn.Module, FrozenDict[str, Any]]) -> FlaxPredictor:
        """Create a Flax predictor for testing."""
        model, params = flax_model_and_params
        return FlaxPredictor(model, params)

    def test_apsscore_with_flax_model(self, flax_predictor: FlaxPredictor) -> None:
        """Test APSScore with Flax model."""
        score = APSScore(model=flax_predictor, randomize=False, random_state=42)

        # Create test data
        rng = np.random.default_rng(42)
        x_calib = rng.random((30, 10), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=30)

        # Test calibration scores
        cal_scores = score.calibration_nonconformity(x_calib, y_calib)

        assert isinstance(cal_scores, np.ndarray)
        assert cal_scores.shape == (30,)

        # Test prediction scores
        x_test = rng.random((10, 10), dtype=np.float32)
        pred_scores = score.predict_nonconformity(x_test)

        assert isinstance(pred_scores, np.ndarray)
        assert pred_scores.shape == (10, 3)

    def test_apsscore_integration(self, flax_predictor: FlaxPredictor) -> None:
        """Test APSScore integrated in SplitConformalPredictor."""
        score = APSScore(model=flax_predictor, randomize=False)
        cp_predictor = SplitConformalPredictor(model=flax_predictor, score=score)

        # Create test data
        rng = np.random.default_rng(42)
        x_cal = rng.random((50, 10), dtype=np.float32)
        y_cal = rng.integers(0, 3, size=50)

        # Calibrate
        threshold = cp_predictor.calibrate(x_cal, y_cal, alpha=0.1)

        assert cp_predictor.is_calibrated
        assert 0 <= threshold <= 1

        # Predict
        x_test = rng.random((10, 10), dtype=np.float32)
        prediction_sets = cp_predictor.predict(x_test, alpha=0.1)

        assert isinstance(prediction_sets, np.ndarray)
        assert prediction_sets.dtype == bool
        assert prediction_sets.shape == (10, 3)

    def test_with_different_random_states(self, flax_predictor: FlaxPredictor) -> None:
        """Test reproducibility with different random states."""
        # Create two scores with same random state
        score1 = APSScore(model=flax_predictor, randomize=True, random_state=42)
        score2 = APSScore(model=flax_predictor, randomize=True, random_state=42)

        rng = np.random.default_rng(42)
        x_data = rng.random((20, 10), dtype=np.float32)
        y_data = rng.integers(0, 3, size=20)

        # Should give same results with same random state
        scores1 = score1.calibration_nonconformity(x_data, y_data)
        scores2 = score2.calibration_nonconformity(x_data, y_data)

        assert np.allclose(scores1, scores2)

        # Different random state should give different results
        score3 = APSScore(model=flax_predictor, randomize=True, random_state=123)
        scores3 = score3.calibration_nonconformity(x_data, y_data)

        # With randomization, they should be different
        assert not np.allclose(scores1, scores3)

    def test_with_and_without_randomization(self, flax_predictor: FlaxPredictor) -> None:
        """Compare scores with and without randomization."""
        score_no_rand = APSScore(model=flax_predictor, randomize=False, random_state=42)
        score_with_rand = APSScore(model=flax_predictor, randomize=True, random_state=42)

        rng = np.random.default_rng(42)
        x_data = rng.random((10, 10), dtype=np.float32)
        y_data = rng.integers(0, 3, size=10)

        scores_no_rand = score_no_rand.calibration_nonconformity(x_data, y_data)
        scores_with_rand = score_with_rand.calibration_nonconformity(x_data, y_data)

        # With randomization enabled, scores should be different
        # (though they could theoretically be the same by chance)
        assert not np.array_equal(scores_no_rand, scores_with_rand)

        # But both should be in valid range
        assert np.all(scores_no_rand >= 0)
        assert np.all(scores_no_rand <= 1)
        assert np.all(scores_with_rand >= 0)
        assert np.all(scores_with_rand <= 1)
