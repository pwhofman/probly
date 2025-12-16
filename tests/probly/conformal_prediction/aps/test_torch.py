"""Tests for torch APS."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from probly.conformal_prediction.aps.methods.split_conformal import SplitConformal
from probly.conformal_prediction.aps.torch import APSPredictor

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

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model() -> nn.Module:
    """Ficture to create a simple PyTorch model."""
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


class TestAPSPredictorTorch:
    """Test class for APSPredictor using PyTorch."""

    def test_initialization(self, simple_model: nn.Module) -> None:
        """Test initialization of APSPredictor."""
        predictor = APSPredictor(model=simple_model)

        assert predictor.model is simple_model
        assert not predictor.is_calibrated
        assert predictor.threshold is None
        assert predictor.nonconformity_scores is None

        # check device
        assert predictor.device.type in ["cpu", "cuda"]

    def test_initialization_with_device(self, simple_model: nn.Module) -> None:
        """Test initialization of APSPredictor with specified device."""
        predictor = APSPredictor(model=simple_model, device="cpu")
        assert predictor.device.type == "cpu"

    def test_calibration(self, simple_model: nn.Module) -> None:
        """Test calibration functionality."""
        predictor = APSPredictor(model=simple_model)

        # create calibration data
        rng = np.random.default_rng(42)
        x_calib = rng.random((30, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=30)

        # calibrate
        significance = 0.1
        threshold = predictor.calibrate(x_calib, y_calib, significance)

        assert predictor.is_calibrated
        assert predictor.threshold == threshold
        assert predictor.nonconformity_scores is not None
        assert len(predictor.nonconformity_scores) == len(x_calib)
        assert 0 <= predictor.threshold <= 1

    def test_prediction_after_calibration(self, simple_model: nn.Module) -> None:
        """Test prediction functionality after calibration."""
        predictor = APSPredictor(model=simple_model)

        # calibrate
        rng = np.random.default_rng(42)
        x_calib = rng.random((20, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=20)
        predictor.calibrate(x_calib, y_calib, significance=0.1)

        # predict
        x_test = rng.random((5, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, significance=0.1)

        # check predictions
        assert len(prediction_sets) == len(x_test)
        assert all(isinstance(pred_set, list) for pred_set in prediction_sets)
        assert all(len(pred_set) >= 1 for pred_set in prediction_sets)

    def test_predictsets_are_valid(self, simple_model: nn.Module) -> None:
        """Test that prediction sets are valid."""
        predictor = APSPredictor(model=simple_model)

        # calibrate
        rng = np.random.default_rng(42)
        x_calib = rng.random((25, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=25)
        predictor.calibrate(x_calib, y_calib, significance=0.1)

        # predict
        x_test = rng.random((10, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, significance=0.1)

        # verify all sets contain valid class indices (0, 1 or 2)
        for pred_set in prediction_sets:
            assert all(0 <= idx < 3 for idx in pred_set)

    def test_str_representation(self, simple_model: nn.Module) -> None:
        """Test string representation of APSPredictor."""
        predictor = APSPredictor(model=simple_model)

        # before calibration
        assert "not calibrated" in str(predictor)
        assert simple_model.__class__.__name__ in str(predictor)

        # after calibration
        rng = np.random.default_rng(42)
        x_calib = rng.random((10, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=10)
        predictor.calibrate(x_calib, y_calib, significance=0.1)

        assert "calibrated" in str(predictor)

    def test_integration_with_split_conformal(self, simple_model: nn.Module) -> None:
        """Test integration with split conformal."""
        predictor = APSPredictor(model=simple_model)

        # create full dataset
        rng = np.random.default_rng(42)
        x_full = rng.random((150, 5), dtype=np.float32)
        y_full = rng.integers(0, 3, size=150)

        # use fit_with_split
        splitter = SplitConformal(calibration_ratio=0.3, random_state=42)

        predictor.set_splitter(splitter)

        x_train, y_train = predictor.fit_with_split(
            x_full,
            y_full,
            significance_level=0.1,
            calibration_ratio=0.3,
        )

        # verify split
        assert predictor.nonconformity_scores is not None
        assert len(x_train) + len(predictor.nonconformity_scores) == len(x_full)
        assert predictor.is_calibrated
        assert predictor.threshold is not None

        # make predictions
        x_test = rng.random((10, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, significance=0.1)

        assert len(prediction_sets) == len(x_test)

    def test_with_iris(self) -> None:
        """Test APSPredictor with IRIS dataset."""
        if not HAS_SKLEARN:
            pytest.skip("sklearn is not installed")

        # load data
        iris = load_iris()
        x, y = iris.data, iris.target

        # split data into train, calibration, and test sets
        x_temp, x_test, y_temp, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # split temp into train and calibration
        x_train, x_calib, y_train, y_calib = train_test_split(
            x_temp,
            y_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_temp,
        )

        # scale features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
        x_calib_scaled = scaler.transform(x_calib).astype(np.float32)
        x_test_scaled = scaler.transform(x_test).astype(np.float32)

        # Update simple model to match IRIS data
        model = SimpleNet(input_dim=4, output_dim=3)

        # train model
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # convert to tensors
        x_train_tensor = torch.tensor(x_train_scaled)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        # simple training loop
        for _ in range(50):
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # initialize predictor
        predictor = APSPredictor(model=model)

        # calibrate
        significance = 0.1
        threshold = predictor.calibrate(x_calib_scaled, y_calib, significance)

        # check calibration
        assert predictor.is_calibrated
        assert predictor.threshold == threshold
        assert 0 <= predictor.threshold <= 1

        # predict
        prediction_sets = predictor.predict(x_test_scaled, significance)

        # check predictions
        assert len(prediction_sets) == len(x_test)
        assert all(isinstance(pred_set, list) for pred_set in prediction_sets)

        assert all(len(pred_set) >= 1 for pred_set in prediction_sets)

        # check all predicted classes are valid (0, 1, 2 for IRIS)
        for pred_set in prediction_sets:
            assert all(0 <= idx < 3 for idx in pred_set)

        # calculate coverage
        correct = 0
        for i, true_label in enumerate(y_test):
            if true_label in prediction_sets[i]:
                correct += 1

        coverage = correct / len(y_test)

        # coevrage should be close to 1- significance = 0.9
        assert 0.8 <= coverage <= 1.0
