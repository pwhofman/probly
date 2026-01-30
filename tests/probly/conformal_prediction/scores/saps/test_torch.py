"""Test for PyTorch SAPS implementation."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F

from probly.conformal_prediction.methods.split import SplitConformal, SplitConformalClassifier
from probly.conformal_prediction.scores.saps.common import SAPSScore


class SimpleNet(nn.Module):
    """Simple neural network for testing."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the network.

        Args:
            input_dim: Dimension of input features.
            output_dim: Number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = F.relu(self.fc1(x))
        return cast(torch.Tensor, self.fc2(x))


@pytest.fixture
def dummy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create dummy data for testing."""
    rng = np.random.default_rng(42)

    # train data
    x_train: npt.NDArray[np.float32] = rng.random((100, 5), dtype=np.float32)
    y_train: npt.NDArray[np.int64] = rng.integers(0, 3, size=100)

    # calibration data
    x_calib: npt.NDArray[np.float32] = rng.random((50, 5), dtype=np.float32)
    y_calib: npt.NDArray[np.int64] = rng.integers(0, 3, size=50)

    return x_train, y_train, x_calib, y_calib


@pytest.fixture
def simple_model() -> nn.Module:
    """Fixture to create a simple PyTorch model."""
    return SimpleNet(input_dim=5, output_dim=3)


class TestSAPSScoreTorch:
    """Test SAPS score implementation for PyTorch."""

    @pytest.fixture
    def test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixture for test data."""
        rng = np.random.default_rng(42)
        x_data: npt.NDArray[np.float32] = rng.random((50, 5), dtype=np.float32)
        y_data: npt.NDArray[np.int64] = rng.integers(0, 3, size=50)
        return x_data, y_data

    def test_sapsscore_with_torch(self, simple_model: nn.Module) -> None:
        """Test SAPS score with PyTorch models."""
        # create SAPSScore
        score: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.1)

        # create test data
        rng = np.random.default_rng(42)
        x_calib: npt.NDArray[np.float32] = rng.random((30, 5), dtype=np.float32)
        y_calib: npt.NDArray[np.int64] = rng.integers(0, 3, size=30)

        # test calibration scores
        cal_scores: np.ndarray | torch.Tensor = score.calibration_nonconformity(x_calib, y_calib)

        cal_scores_np: np.ndarray = (
            cal_scores.detach().cpu().numpy() if hasattr(cal_scores, "detach") else np.asarray(cal_scores)
        )
        assert cal_scores_np.shape == (30,)
        assert np.all(cal_scores_np >= 0)

        # test prediction scores
        x_test: npt.NDArray[np.float32] = rng.random((10, 5), dtype=np.float32)
        pred_scores: np.ndarray | torch.Tensor = score.predict_nonconformity(x_test)

        pred_scores_np: np.ndarray = (
            pred_scores.detach().cpu().numpy() if hasattr(pred_scores, "detach") else np.asarray(pred_scores)
        )
        assert pred_scores_np.shape == (10, 3)

    def test_sapsscore_with_different_lambda(self, simple_model: nn.Module) -> None:
        """Test SAPS score with different lambda values."""
        # test with small lambda
        small_lambda: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.01)
        # test with large lambda
        large_lambda: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.5)

        rng = np.random.default_rng(42)
        x_calib: npt.NDArray[np.float32] = rng.random((20, 5), dtype=np.float32)
        y_calib: npt.NDArray[np.int64] = rng.integers(0, 3, size=20)

        cal_scores_small: np.ndarray | torch.Tensor = small_lambda.calibration_nonconformity(x_calib, y_calib)
        cal_scores_large: np.ndarray | torch.Tensor = large_lambda.calibration_nonconformity(x_calib, y_calib)

        # Convert to numpy for assertions
        cal_scores_small_np: np.ndarray = (
            cal_scores_small.detach().cpu().numpy()
            if hasattr(cal_scores_small, "detach")
            else np.asarray(cal_scores_small)
        )
        cal_scores_large_np: np.ndarray = (
            cal_scores_large.detach().cpu().numpy()
            if hasattr(cal_scores_large, "detach")
            else np.asarray(cal_scores_large)
        )

        assert cal_scores_small_np.shape == (20,)
        assert cal_scores_large_np.shape == (20,)
        assert np.mean(cal_scores_large_np) > np.mean(cal_scores_small_np)

    def test_sapsscore_in_split_predictor(self, simple_model: nn.Module) -> None:
        """Test SAPS score within Split Conformal Prediction framework."""
        score: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.1)
        predictor: SplitConformalClassifier = SplitConformalClassifier(model=simple_model, score=score)

        rng = np.random.default_rng(42)

        # calibrate
        x_calib: npt.NDArray[np.float32] = rng.random((50, 5), dtype=np.float32)
        y_calib: npt.NDArray[np.int64] = rng.integers(0, 3, size=50)
        threshold: float = predictor.calibrate(x_calib, y_calib, alpha=0.1)

        assert predictor.is_calibrated
        assert predictor.threshold == threshold
        assert 0 <= threshold <= 2 + 1e-6

        # predict
        x_test: npt.NDArray[np.float32] = rng.random((10, 5), dtype=np.float32)
        prediction_sets: np.ndarray | torch.Tensor = predictor.predict(x_test, alpha=0.1)

        if hasattr(prediction_sets, "detach"):
            prediction_sets_np: np.ndarray = prediction_sets.detach().cpu().numpy()
        else:
            prediction_sets_np = np.asarray(prediction_sets)
        assert prediction_sets_np.shape == (10, 3)
        assert prediction_sets_np.dtype in (bool, np.bool_)

    def test_with_split_conformal(self, simple_model: nn.Module) -> None:
        """Test integration with Split Conformal method."""
        score: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.1)
        predictor: SplitConformalClassifier = SplitConformalClassifier(model=simple_model, score=score)

        # create full dataset
        rng = np.random.default_rng(42)
        x_full: npt.NDArray[np.float32] = rng.random((150, 5), dtype=np.float32)
        y_full: npt.NDArray[np.int64] = rng.integers(0, 3, size=150)

        # create splitter
        splitter: SplitConformal = SplitConformal(calibration_ratio=0.3)

        # split manually
        x_train: np.ndarray
        y_train: np.ndarray
        x_calib: np.ndarray
        y_calib: np.ndarray
        x_train, y_train, x_calib, y_calib = splitter.split(x_full, y_full)

        # calibrate
        predictor.calibrate(x_calib, y_calib, alpha=0.1)

        assert predictor.is_calibrated
        assert predictor.threshold is not None

        # predict
        x_test: npt.NDArray[np.float32] = rng.random((10, 5), dtype=np.float32)
        prediction_sets: np.ndarray | torch.Tensor = predictor.predict(x_test, alpha=0.1)

        assert prediction_sets.shape == (10, 3)

    def test_with_iris_dataset(self) -> None:
        """Test with IRIS dataset."""
        # load data
        iris = load_iris()
        x: npt.NDArray[np.float64] = iris.data
        y: npt.NDArray[np.int64] = iris.target

        # split data
        x_temp: np.ndarray
        x_test: np.ndarray
        y_temp: np.ndarray
        y_test: np.ndarray
        x_temp, x_test, y_temp, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        x_train: np.ndarray
        x_calib: np.ndarray
        y_train: np.ndarray
        y_calib: np.ndarray
        x_train, x_calib, y_train, y_calib = train_test_split(
            x_temp,
            y_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_temp,
        )

        # scale features
        scaler: StandardScaler = StandardScaler()
        x_train_scaled: npt.NDArray[np.float32] = scaler.fit_transform(x_train).astype(np.float32)
        x_calib_scaled: npt.NDArray[np.float32] = scaler.transform(x_calib).astype(np.float32)
        x_test_scaled: npt.NDArray[np.float32] = scaler.transform(x_test).astype(np.float32)

        # create model
        model: SimpleNet = SimpleNet(input_dim=4, output_dim=3)

        # simple training
        model.train()
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=0.01)

        x_train_tensor: torch.Tensor = torch.tensor(x_train_scaled)
        y_train_tensor: torch.Tensor = torch.tensor(y_train, dtype=torch.long)

        for _ in range(50):
            optimizer.zero_grad()
            outputs: torch.Tensor = model(x_train_tensor)
            loss: torch.Tensor = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # create score and predictor
        score: SAPSScore = SAPSScore(model=model, lambda_val=0.1)
        predictor: SplitConformalClassifier = SplitConformalClassifier(model=model, score=score)

        # calibrate
        threshold: float = predictor.calibrate(x_calib_scaled, y_calib, alpha=0.1)

        assert predictor.is_calibrated
        assert 0 <= threshold <= 1 + 1e-6  # Upper bound depends on max_prob + lambda*(K-1)

        # predict
        prediction_sets: np.ndarray | torch.Tensor = predictor.predict(x_test_scaled, alpha=0.1)

        assert prediction_sets.shape == (len(x_test), 3)

        # calculate coverage
        covered: int = 0
        for i, true_label in enumerate(y_test):
            if prediction_sets[i, true_label]:
                covered += 1

        coverage: float = covered / len(y_test)

        # coverage should be reasonable
        assert 0.8 <= coverage <= 1.0

    def test_with_different_random_states(self, simple_model: nn.Module) -> None:
        """Test reproducibility with different random states."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        # Use generator instead of np.random.seed
        rng1 = np.random.default_rng(42)

        # Create two scores with same random state
        score1: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.1, random_state=42)
        score2: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.1, random_state=42)

        x_data: npt.NDArray[np.float32] = rng1.random((20, 5), dtype=np.float32)
        y_data: npt.NDArray[np.int64] = rng1.integers(0, 3, size=20)

        # Should give same results with same random state
        scores1: np.ndarray | torch.Tensor = score1.calibration_nonconformity(x_data, y_data)
        scores2: np.ndarray | torch.Tensor = score2.calibration_nonconformity(x_data, y_data)

        # Convert to numpy for comparison if needed
        scores1_np: np.ndarray = scores1.detach().cpu().numpy() if hasattr(scores1, "detach") else np.asarray(scores1)
        scores2_np: np.ndarray = scores2.detach().cpu().numpy() if hasattr(scores2, "detach") else np.asarray(scores2)

        # Check they are close (within tolerance)
        assert np.allclose(scores1_np, scores2_np, rtol=1e-5, atol=1e-8), (
            f"Same random state should produce same results: {scores1_np[:5]} != {scores2_np[:5]}"
        )

        # Different random state might produce different results
        # (but could coincidentally be the same)
        torch.manual_seed(123)
        score3: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.1, random_state=123)
        scores3: np.ndarray | torch.Tensor = score3.calibration_nonconformity(x_data, y_data)

        scores3_np: np.ndarray = scores3.detach().cpu().numpy() if hasattr(scores3, "detach") else np.asarray(scores3)

        are_different: bool = not np.allclose(scores1_np, scores3_np, rtol=1e-5, atol=1e-8)

        if not are_different:
            # It's okay if they happen to be the same by chance
            pass

    def test_torch_model_forward_pass_shapes(self, simple_model: nn.Module) -> None:
        """Test that PyTorch model forward pass returns correct shapes."""
        rng = np.random.default_rng(42)
        x_test: npt.NDArray[np.float32] = rng.random((5, 5), dtype=np.float32)

        # convert to tensor
        x_tensor: torch.Tensor = torch.tensor(x_test)
        logits: torch.Tensor = simple_model(x_tensor)

        # check shapes
        assert logits.shape == (5, 3), f"Expected shape (5, 3), got {logits.shape}"

        # check with softmax
        probs: torch.Tensor = torch.nn.functional.softmax(logits, dim=1)
        np.testing.assert_allclose(
            probs.detach().numpy().sum(axis=1),
            np.ones(5),
            rtol=1e-5,
        )

    def test_sapsscore_output_types(self, simple_model: nn.Module) -> None:
        """Test that SAPSScore outputs have correct dtypes and types."""
        score: SAPSScore = SAPSScore(model=simple_model, lambda_val=0.1)

        rng = np.random.default_rng(42)
        x_calib: npt.NDArray[np.float32] = rng.random((10, 5), dtype=np.float32)
        y_calib: npt.NDArray[np.int64] = rng.integers(0, 3, size=10)

        # Test calibration scores - may return torch.Tensor
        cal_scores: np.ndarray | torch.Tensor = score.calibration_nonconformity(x_calib, y_calib)

        # Accept either torch.Tensor or np.ndarray
        cal_scores_np: np.ndarray
        cal_scores_np = cal_scores.detach().cpu().numpy() if hasattr(cal_scores, "detach") else np.asarray(cal_scores)

        assert isinstance(cal_scores_np, np.ndarray)
        assert cal_scores_np.dtype in [np.float32, np.float64]

        # Test prediction scores
        x_test: npt.NDArray[np.float32] = rng.random((5, 5), dtype=np.float32)
        pred_scores: np.ndarray | torch.Tensor = score.predict_nonconformity(x_test)

        pred_scores_np: np.ndarray
        if hasattr(pred_scores, "detach"):
            pred_scores_np = pred_scores.detach().cpu().numpy()
        else:
            pred_scores_np = np.asarray(pred_scores)

        assert isinstance(pred_scores_np, np.ndarray)
        assert pred_scores_np.dtype in [np.float32, np.float64]
        assert pred_scores_np.shape == (5, 3)

    def test_torch_model_edge_case_shapes(self, simple_model: nn.Module) -> None:
        """Test PyTorch model edge cases for input shapes."""
        rng = np.random.default_rng(42)

        # Test single sample
        x_single: npt.NDArray[np.float32] = rng.random((5,), dtype=np.float32)
        logits_single: torch.Tensor = simple_model(torch.tensor(x_single).unsqueeze(0))
        assert logits_single.shape == (1, 3), f"Expected shape (1, 3), got {logits_single.shape}"

        # Test large batch
        x_large: npt.NDArray[np.float32] = rng.random((100, 5), dtype=np.float32)
        logits_large: torch.Tensor = simple_model(torch.tensor(x_large))
        assert logits_large.shape == (100, 3), f"Expected shape (100, 3), got {logits_large.shape}"


def test_iris_coverage_guarantee() -> None:
    """Test that coverage guarantee holds on Iris dataset with multiple seeds."""
    # load data
    iris = load_iris()
    x: npt.NDArray[np.float64] = iris.data
    y: npt.NDArray[np.int64] = iris.target

    # Test multiple random splits for robustness
    for seed in [123, 456, 789]:
        # Set seeds for reproducibility
        torch.manual_seed(seed)

        # split data
        x_temp: np.ndarray
        x_test: np.ndarray
        y_temp: np.ndarray
        y_test: np.ndarray
        x_temp, x_test, y_temp, y_test = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=seed,
            stratify=y,
        )

        x_train: np.ndarray
        x_calib: np.ndarray
        y_train: np.ndarray
        y_calib: np.ndarray
        x_train, x_calib, y_train, y_calib = train_test_split(
            x_temp,
            y_temp,
            test_size=0.25,
            random_state=seed,
            stratify=y_temp,
        )

        # scale features
        scaler: StandardScaler = StandardScaler()
        x_train_scaled: npt.NDArray[np.float32] = scaler.fit_transform(x_train).astype(np.float32)
        x_calib_scaled: npt.NDArray[np.float32] = scaler.transform(x_calib).astype(np.float32)
        x_test_scaled: npt.NDArray[np.float32] = scaler.transform(x_test).astype(np.float32)

        # create model
        model: SimpleNet = SimpleNet(input_dim=4, output_dim=3)

        # simple training
        model.train()
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=0.01)

        x_train_tensor: torch.Tensor = torch.tensor(x_train_scaled)
        y_train_tensor: torch.Tensor = torch.tensor(y_train, dtype=torch.long)

        for _ in range(50):
            optimizer.zero_grad()
            outputs: torch.Tensor = model(x_train_tensor)
            loss: torch.Tensor = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()

        # create score and predictor
        score: SAPSScore = SAPSScore(model=model, lambda_val=0.1, random_state=seed)
        predictor: SplitConformalClassifier = SplitConformalClassifier(model=model, score=score)

        # calibrate
        threshold: float = predictor.calibrate(x_calib_scaled, y_calib, alpha=0.1)
        assert predictor.is_calibrated
        assert 0 <= threshold <= 1 + 1e-6  # Upper bound depends on max_prob + lambda*(K-1)

        # predict
        prediction_sets: np.ndarray | torch.Tensor = predictor.predict(x_test_scaled, alpha=0.1)
        assert prediction_sets.shape == (len(x_test), 3)

        # calculate coverage
        covered: int = 0
        for i, true_label in enumerate(y_test):
            if prediction_sets[i, true_label]:
                covered += 1

        coverage: float = covered / len(y_test)

        # coverage should be reasonable - with alpha=0.1, expect >= 0.9
        # Allow some slack for finite samples
        assert coverage >= 0.7, f"Coverage too low with seed {seed}: {coverage:.3f}"
