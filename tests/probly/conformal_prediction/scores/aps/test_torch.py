"""Tests for PyTorch APS implementation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F

from probly.conformal_prediction.methods.split import SplitConformal, SplitConformalPredictor
from probly.conformal_prediction.scores.aps.common import APSScore


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

    @pytest.fixture
    def test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create test data."""
        rng = np.random.default_rng(42)
        x_data = rng.random((50, 5), dtype=np.float32)
        y_data = rng.integers(0, 3, size=50)
        return x_data, y_data

    def test_apsscore_with_torch_model(self, simple_model: nn.Module) -> None:
        """Test APSScore with PyTorch model."""
        # create APSScore with the model
        score = APSScore(model=simple_model, randomize=False, random_state=42)

        # create some test data
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

        # calibrate
        x_cal = rng.random((50, 5), dtype=np.float32)
        y_cal = rng.integers(0, 3, size=50)
        threshold = predictor.calibrate(x_cal, y_cal, alpha=0.1)

        assert predictor.is_calibrated
        assert predictor.threshold == threshold
        assert 0 <= threshold <= 1

        # predict
        x_test = rng.random((10, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, alpha=0.1)

        assert isinstance(prediction_sets, np.ndarray)
        assert prediction_sets.dtype == bool
        assert prediction_sets.shape == (10, 3)

    def test_with_split_conformal(self, simple_model: nn.Module) -> None:
        """Test integration with split conformal."""
        score = APSScore(model=simple_model, randomize=False)
        predictor = SplitConformalPredictor(model=simple_model, score=score)

        # create full dataset
        rng = np.random.default_rng(42)
        x_full = rng.random((150, 5), dtype=np.float32)
        y_full = rng.integers(0, 3, size=150)

        # create splitter
        splitter = SplitConformal(calibration_ratio=0.3, random_state=42)

        # split manually
        x_train, y_train, x_cal, y_cal = splitter.split(x_full, y_full)

        # calibrate
        predictor.calibrate(x_cal, y_cal, alpha=0.1)

        # verify calibration
        assert predictor.is_calibrated
        assert predictor.threshold is not None

        # make predictions
        x_test = rng.random((10, 5), dtype=np.float32)
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

        # scale features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
        x_calib_scaled = scaler.transform(x_calib).astype(np.float32)
        x_test_scaled = scaler.transform(x_test).astype(np.float32)

        # create model
        model = SimpleNet(input_dim=4, output_dim=3)

        # simple training
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

        # create score and predictor
        score = APSScore(model=model, randomize=False, random_state=42)
        predictor = SplitConformalPredictor(model=model, score=score)

        # calibrate
        threshold = predictor.calibrate(x_calib_scaled, y_calib, alpha=0.1)

        assert predictor.is_calibrated
        assert 0 <= threshold <= 1

        # predict
        prediction_sets = predictor.predict(x_test_scaled, alpha=0.1)

        assert prediction_sets.shape == (len(x_test), 3)

        # calculate coverage
        covered = 0
        for i, true_label in enumerate(y_test):
            if prediction_sets[i, true_label]:
                covered += 1

        coverage = covered / len(y_test)

        # coverage should be reasonable
        assert 0.8 <= coverage <= 1.0

    def test_with_different_random_states(self, simple_model: nn.Module) -> None:
        """Test reproducibility with different random states."""
        # create two scores with same random state
        score1 = APSScore(model=simple_model, randomize=True, random_state=42)
        score2 = APSScore(model=simple_model, randomize=True, random_state=42)

        rng = np.random.default_rng(42)
        x_data = rng.random((20, 5), dtype=np.float32)
        y_data = rng.integers(0, 3, size=20)

        # should give same results with same random state
        scores1 = score1.calibration_nonconformity(x_data, y_data)
        scores2 = score2.calibration_nonconformity(x_data, y_data)

        assert np.allclose(scores1, scores2)

        # different random state should give different results
        score3 = APSScore(model=simple_model, randomize=True, random_state=123)
        scores3 = score3.calibration_nonconformity(x_data, y_data)

        # with randomization, they should be different
        assert not np.allclose(scores1, scores3)

    def test_with_and_without_randomization(self, simple_model: nn.Module) -> None:
        """Compare scores with and without randomization."""
        score_no_rand = APSScore(model=simple_model, randomize=False, random_state=42)
        score_with_rand = APSScore(model=simple_model, randomize=True, random_state=42)

        rng = np.random.default_rng(42)
        x_data = rng.random((10, 5), dtype=np.float32)
        y_data = rng.integers(0, 3, size=10)

        scores_no_rand = score_no_rand.calibration_nonconformity(x_data, y_data)
        scores_with_rand = score_with_rand.calibration_nonconformity(x_data, y_data)

        # with randomization enabled, scores should be different
        assert not bool(np.array_equal(scores_no_rand, scores_with_rand))

        # both should be in valid range (allow tolerance for float32 precision)
        assert bool(np.all(scores_no_rand >= 0))
        assert bool(np.all(scores_no_rand <= 1 + 1e-6))
        assert bool(np.all(scores_with_rand >= 0))
        assert bool(np.all(scores_with_rand <= 1 + 1e-6))

    def test_torch_model_forward_pass_shapes(self, simple_model: nn.Module) -> None:
        """Test that PyTorch model forward pass returns correct shapes."""
        rng = np.random.default_rng(42)
        x_test = rng.random((5, 5), dtype=np.float32)

        # convert to tensor
        x_tensor = torch.tensor(x_test)
        logits = simple_model(x_tensor)

        # check shapes
        assert logits.shape == (5, 3), f"Expected shape (5, 3), got {logits.shape}"

        # check with softmax
        probs = torch.nn.functional.softmax(logits, dim=1)
        np.testing.assert_allclose(
            probs.detach().numpy().sum(axis=1),
            np.ones(5),
            rtol=1e-5,
        )

    def test_apsscore_output_types(self, simple_model: nn.Module) -> None:
        """Test that APSScore outputs have correct dtypes and types."""
        score = APSScore(model=simple_model, randomize=False, random_state=42)

        rng = np.random.default_rng(42)
        x_calib = rng.random((10, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=10)

        # Test calibration scores
        cal_scores = score.calibration_nonconformity(x_calib, y_calib)

        assert isinstance(cal_scores, np.ndarray), f"Expected np.ndarray, got {type(cal_scores)}"
        assert cal_scores.dtype in [np.float32, np.float64], f"Expected float dtype, got {cal_scores.dtype}"

        # Test prediction scores
        x_test = rng.random((5, 5), dtype=np.float32)
        pred_scores = score.predict_nonconformity(x_test)

        assert isinstance(pred_scores, np.ndarray), f"Expected np.ndarray, got {type(pred_scores)}"
        assert pred_scores.dtype in [np.float32, np.float64], f"Expected float dtype, got {pred_scores.dtype}"
        assert pred_scores.shape == (5, 3), f"Expected shape (5, 3), got {pred_scores.shape}"

    def test_torch_model_edge_case_shapes(self, simple_model: nn.Module) -> None:
        """Test PyTorch model edge cases for input shapes."""
        rng = np.random.default_rng(42)

        # Test single sample
        x_single = rng.random((5,), dtype=np.float32)
        logits_single = simple_model(torch.tensor(x_single).unsqueeze(0))
        assert logits_single.shape == (1, 3), f"Expected shape (1, 3), got {logits_single.shape}"

        # Test large batch
        x_large = rng.random((100, 5), dtype=np.float32)
        logits_large = simple_model(torch.tensor(x_large))
        assert logits_large.shape == (100, 3), f"Expected shape (100, 3), got {logits_large.shape}"

    def test_iris_coverage_guarantee(self) -> None:
        """Test that coverage guarantee holds on Iris dataset with multiple seeds."""
        # load data
        iris = load_iris()
        x, y = iris.data, iris.target

        # Test multiple random splits for robustness
        for seed in [123, 456, 789]:
            # split data
            x_temp, x_test, y_temp, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                random_state=seed,
                stratify=y,
            )

            x_train, x_calib, y_train, y_calib = train_test_split(
                x_temp,
                y_temp,
                test_size=0.25,
                random_state=seed,
                stratify=y_temp,
            )

            # scale features
            scaler = StandardScaler()
            x_calib_scaled = scaler.fit_transform(x_calib).astype(np.float32)
            x_test_scaled = scaler.transform(x_test).astype(np.float32)

            # create model with specific seed
            model = SimpleNet(input_dim=4, output_dim=3)

            # create score and predictor
            score = APSScore(model=model, randomize=False, random_state=seed)
            predictor = SplitConformalPredictor(model=model, score=score)

            # calibrate
            threshold = predictor.calibrate(x_calib_scaled, y_calib, alpha=0.1)

            assert predictor.is_calibrated
            assert 0 <= threshold <= 1

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
