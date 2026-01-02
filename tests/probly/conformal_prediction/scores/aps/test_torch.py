"""Tests for PyTorch APS implementation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from probly.conformal_prediction.methods.split import SplitConformal, SplitConformalPredictor
from probly.conformal_prediction.scores.aps.common import APSScore

try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SimpleNet(nn.Module):
    """A simple neural network for testing."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the simple neural network.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return cast(torch.Tensor, self.fc2(x))


@pytest.fixture
def simple_model() -> nn.Module:
    """Fixture to create a simple PyTorch model."""
    return SimpleNet(input_dim=5, output_dim=3)


@pytest.fixture
def dummy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate dummy data for testing."""
    rng = np.random.default_rng(42)

    # train data
    x_train = rng.random((100, 5), dtype=np.float32)
    y_train = rng.integers(0, 3, size=100)

    # calibration data
    x_calib = rng.random((50, 5), dtype=np.float32)
    y_calib = rng.integers(0, 3, size=50)

    return x_train, y_train, x_calib, y_calib


class TestAPSScoreTorch:
    """Test class for APSScore using PyTorch models."""

    def test_apsscore_with_torch_model(self, simple_model: nn.Module) -> None:
        """Test APSScore with PyTorch model."""
        # Create APSScore with the model
        score = APSScore(model=simple_model, randomize=False, random_state=42)

        # Create some test data
        rng = np.random.default_rng(42)
        x_calib = rng.random((30, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=30)

        # Test calibration scores
        cal_scores = score.calibration_nonconformity(x_calib, y_calib)

        assert isinstance(cal_scores, np.ndarray)
        assert cal_scores.shape == (30,)
        assert np.all(cal_scores >= 0)

        # Test prediction scores
        x_test = rng.random((10, 5), dtype=np.float32)
        pred_scores = score.predict_nonconformity(x_test)

        assert isinstance(pred_scores, np.ndarray)
        assert pred_scores.shape == (10, 3)  # 10 samples, 3 classes

    def test_apsscore_with_randomization(self, simple_model: nn.Module) -> None:
        """Test APSScore with randomization enabled."""
        score = APSScore(model=simple_model, randomize=True, random_state=42)

        rng = np.random.default_rng(42)
        x_calib = rng.random((20, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=20)

        cal_scores = score.calibration_nonconformity(x_calib, y_calib)

        assert cal_scores.shape == (20,)
        assert np.all(cal_scores <= 1.0)  # APS scores should be <= 1

    def test_apsscore_in_split_predictor(self, simple_model: nn.Module) -> None:
        """Test APSScore integrated in SplitConformalPredictor."""
        score = APSScore(model=simple_model, randomize=False)
        predictor = SplitConformalPredictor(model=simple_model, score=score)

        rng = np.random.default_rng(42)

        # Calibrate
        x_cal = rng.random((50, 5), dtype=np.float32)
        y_cal = rng.integers(0, 3, size=50)
        threshold = predictor.calibrate(x_cal, y_cal, alpha=0.1)

        assert predictor.is_calibrated
        assert predictor.threshold == threshold
        assert 0 <= threshold <= 1

        # Predict
        x_test = rng.random((10, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, alpha=0.1)

        assert isinstance(prediction_sets, np.ndarray)
        assert prediction_sets.dtype == bool
        assert prediction_sets.shape == (10, 3)

    def test_with_split_conformal(self, simple_model: nn.Module) -> None:
        """Test integration with split conformal."""
        score = APSScore(model=simple_model, randomize=False)
        predictor = SplitConformalPredictor(model=simple_model, score=score)

        # Create full dataset
        rng = np.random.default_rng(42)
        x_full = rng.random((150, 5), dtype=np.float32)
        y_full = rng.integers(0, 3, size=150)

        # Create splitter
        splitter = SplitConformal(calibration_ratio=0.3, random_state=42)

        # Split manually
        x_train, y_train, x_cal, y_cal = splitter.split(x_full, y_full)

        # Calibrate
        predictor.calibrate(x_cal, y_cal, alpha=0.1)

        # Verify calibration
        assert predictor.is_calibrated
        assert predictor.threshold is not None

        # Make predictions
        x_test = rng.random((10, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, alpha=0.1)

        assert prediction_sets.shape == (10, 3)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn is not installed")
    def test_with_iris_dataset(self) -> None:
        """Test with real Iris dataset."""
        # Load data
        iris = load_iris()
        x, y = iris.data, iris.target

        # Split data
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

        # Scale features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
        x_calib_scaled = scaler.transform(x_calib).astype(np.float32)
        x_test_scaled = scaler.transform(x_test).astype(np.float32)

        # Create model
        model = SimpleNet(input_dim=4, output_dim=3)

        # Simple training
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        x_train_tensor = torch.tensor(x_train_scaled)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        for _ in range(50):
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Create score and predictor
        score = APSScore(model=model, randomize=False, random_state=42)
        predictor = SplitConformalPredictor(model=model, score=score)

        # Calibrate
        threshold = predictor.calibrate(x_calib_scaled, y_calib, alpha=0.1)

        assert predictor.is_calibrated
        assert 0 <= threshold <= 1

        # Predict
        prediction_sets = predictor.predict(x_test_scaled, alpha=0.1)

        assert prediction_sets.shape == (len(x_test), 3)

        # Calculate coverage
        covered = 0
        for i, true_label in enumerate(y_test):
            if prediction_sets[i, true_label]:
                covered += 1

        coverage = covered / len(y_test)

        # Coverage should be reasonable
        assert 0.8 <= coverage <= 1.0
