"""Tests for PyTorch LAC implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim

from probly.conformal_prediction.methods.common import predict_probs
from probly.conformal_prediction.methods.split import SplitConformalClassifier
from probly.conformal_prediction.scores.lac.common import LACScore


class MockTorchModel(nn.Module):
    """Mock PyTorch model for testing."""

    def __init__(self, n_classes: int = 3, true_prob: float = 0.9) -> None:
        """Initialize mock PyTorch model."""
        super().__init__()
        self.n_classes = n_classes
        self.true_prob = true_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        device = x.device

        # Create probability distribution
        probs = torch.zeros(self.n_classes, device=device)
        probs[0] = self.true_prob

        if self.n_classes > 1:
            remaining = (1.0 - self.true_prob) / (self.n_classes - 1)
            probs[1:] = remaining

        return probs.repeat(n_samples, 1)

    def predict(self, x: Sequence[Any]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return self.forward(x)
        return self.forward(torch.tensor(x, dtype=torch.float32))


class SimpleTorchModel(nn.Module):
    """Simple trainable PyTorch model for real data testing."""

    def __init__(self, input_dim: int = 4, n_classes: int = 3) -> None:
        """Initialize simple PyTorch model."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax output."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        result: torch.Tensor = self.softmax(logits)
        return result

    def predict(self, x: Sequence[Any]) -> torch.Tensor:
        """Predict probabilities."""
        if isinstance(x, torch.Tensor):
            return self.forward(x).detach()
        return self.forward(torch.tensor(x, dtype=torch.float32)).detach()


@predict_probs.register(MockTorchModel)
def predict_probs_mock_torch(model: MockTorchModel, x: Sequence[Any]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return model.forward(x)
    return model.forward(torch.tensor(x, dtype=torch.float32))


@predict_probs.register(SimpleTorchModel)
def predict_probs_simple_torch(model: SimpleTorchModel, x: Sequence[Any]) -> torch.Tensor:
    """Predict probabilities for simple model."""
    result: torch.Tensor = model.predict(x)
    return result


def test_lacscore_with_torch_model() -> None:
    """Test LACScore with PyTorch model."""
    model = MockTorchModel(true_prob=0.9)
    score = LACScore(model=model)

    # Create test data
    rng = np.random.default_rng(42)
    x_calib = rng.random((30, 5), dtype=np.float32)
    y_calib = np.zeros(30, dtype=int)  # All class 0

    # Test calibration scores
    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (30,)
    # LAC score = 1 - probability of true class
    # With true_prob=0.9, expected score = 0.1
    assert np.allclose(cal_scores, 0.1, atol=0.1)

    # Test prediction scores
    x_test = rng.random((10, 5), dtype=np.float32)
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (10, 3)  # 10 samples, 3 classes
    # For mock model: probabilities [0.9, 0.05, 0.05] -> scores = 1 - p
    expected_scores = np.array([[0.1, 0.95, 0.95]] * 10)
    assert np.allclose(pred_scores, expected_scores, atol=0.1)


def test_lacscore_in_split_predictor() -> None:
    """Test LACScore integrated in SplitConformalClassifier."""
    model = MockTorchModel(true_prob=0.8)
    score = LACScore(model=model)
    predictor = SplitConformalClassifier(model=model, score=score, use_accretive=True)

    rng = np.random.default_rng(42)

    # Calibrate
    x_cal = rng.random((50, 5), dtype=np.float32)
    y_cal = np.zeros(50, dtype=int)  # All class 0
    threshold = predictor.calibrate(x_cal, y_cal, alpha=0.1)

    assert predictor.is_calibrated
    assert predictor.threshold == threshold
    # With true_prob=0.8, LAC scores = 0.2, quantile should be ~0.2
    assert 0.19 <= threshold <= 0.21

    # Predict
    x_test = rng.random((10, 5), dtype=np.float32)
    prediction_sets = predictor.predict(x_test, alpha=0.1)

    assert isinstance(prediction_sets, np.ndarray)
    assert prediction_sets.dtype == bool
    assert prediction_sets.shape == (10, 3)

    # With threshold ~0.2 and scores: class0=0.2, others=0.6
    # class0 should be included (score <= threshold)
    # class1,2 should not be included (score > threshold)
    assert np.all(prediction_sets[:, 0])  # All should include class 0
    assert not np.any(prediction_sets[:, 1])  # None should include class 1
    assert not np.any(prediction_sets[:, 2])  # None should include class 2


def test_lacscore_accretive_completion_comparison() -> None:
    """Test that accretive completion prevents empty prediction sets."""
    # create a model that gives very low probabilities for all classes
    # this can cause empty prediction sets without accretive completion
    model = MockTorchModel(true_prob=0.1)  # very low true class probability
    score = LACScore(model=model)

    rng = np.random.default_rng(42)
    x_cal = rng.random((50, 5), dtype=np.float32)
    y_cal = rng.integers(0, 3, size=50)

    # Test without accretive completion
    predictor_no_accretive = SplitConformalClassifier(
        model=model,
        score=score,
        use_accretive=False,
    )
    predictor_no_accretive.calibrate(x_cal, y_cal, alpha=0.1)
    x_test = rng.random((20, 5), dtype=np.float32)
    sets_no_accretive = predictor_no_accretive.predict(x_test, alpha=0.1)

    # Test with accretive completion
    predictor_with_accretive = SplitConformalClassifier(
        model=model,
        score=score,
        use_accretive=True,
    )
    predictor_with_accretive.calibrate(x_cal, y_cal, alpha=0.1)
    sets_with_accretive = predictor_with_accretive.predict(x_test, alpha=0.1)

    # without accretive: might have empty sets
    set_sizes_no_accretive = np.sum(sets_no_accretive, axis=1)
    has_empty_sets = np.any(set_sizes_no_accretive == 0)

    # with accretive: all sets should be non-empty
    set_sizes_with_accretive = np.sum(sets_with_accretive, axis=1)
    assert np.all(set_sizes_with_accretive >= 1), "accretive completion should prevent empty sets"

    # if there were empty sets without accretive, verify accretive fixed them
    if has_empty_sets:
        assert set_sizes_with_accretive.min() > set_sizes_no_accretive.min(), (
            "accretive should add elements to empty sets"
        )


def test_lacscore_edge_case_single_sample() -> None:
    """Test LACScore with single sample."""
    model = MockTorchModel()
    score = LACScore(model=model)
    x_calib = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    y_calib = np.array([0])

    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert cal_scores.shape == (1,), f"expected shape (1,), got {cal_scores.shape}"
    assert np.all(cal_scores >= 0), "scores should be non-negative"
    assert np.all(cal_scores <= 1), "scores should be at most 1"


def test_lacscore_edge_case_large_batch() -> None:
    """Test LACScore with large batch."""
    model = MockTorchModel()
    score = LACScore(model=model)
    rng = np.random.default_rng(42)
    x_calib = rng.random((500, 5), dtype=np.float32)
    y_calib = rng.integers(0, 3, size=500)

    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert cal_scores.shape == (500,), f"expected shape (500,), got {cal_scores.shape}"
    assert np.all(cal_scores >= 0), "scores should be non-negative"
    assert np.all(cal_scores <= 1), "scores should be at most 1"


def test_lacscore_output_types() -> None:
    """Test LACScore output types and dtypes."""
    model = MockTorchModel()
    score = LACScore(model=model)
    rng = np.random.default_rng(42)
    x_test = rng.random((10, 5), dtype=np.float32)

    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray), f"expected np.ndarray, got {type(pred_scores)}"
    assert pred_scores.dtype in [np.float32, np.float64], f"expected float dtype, got {pred_scores.dtype}"
    assert pred_scores.shape == (10, 3), f"expected shape (10, 3), got {pred_scores.shape}"


def test_lacscore_torch_model_forward_pass_shapes() -> None:
    """Test torch model forward pass shapes."""
    model = MockTorchModel()
    x_test = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

    output = model.predict(x_test.tolist())

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3), f"expected shape (1, 3), got {output.shape}"


def test_lacscore_multiple_classes() -> None:
    """Test LACScore with different number of classes."""
    model = MockTorchModel(n_classes=3)
    score = LACScore(model=model)
    rng = np.random.default_rng(42)

    # test with different label values
    x_calib = rng.random((20, 5), dtype=np.float32)
    y_calib = rng.integers(0, 3, size=20)

    cal_scores = score.calibration_nonconformity(x_calib, y_calib)

    assert cal_scores.shape == (20,), f"expected shape (20,), got {cal_scores.shape}"
    assert np.all(cal_scores >= 0), "scores should be non-negative"
    assert np.all(cal_scores <= 1), "scores should be at most 1"


def test_lacscore_with_different_label_values() -> None:
    """Test LACScore with different label values."""
    model = MockTorchModel()
    score = LACScore(model=model)

    x_cal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
    # different label values
    y_cal = np.array([0, 2])

    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert cal_scores.shape == (2,)
    assert np.all(cal_scores >= 0)
    assert np.all(cal_scores <= 1)


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
    model = SimpleTorchModel(input_dim=4, n_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # convert data to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # training loop
    for _ in range(100):
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # disable gradients for evaluation
    model.eval()

    # create predictor with trained model
    score = LACScore(model=model)

    # create conformal predictor
    cp_predictor = SplitConformalClassifier(
        model=model,
        score=score,
        use_accretive=True,
    )

    # calibrate with alpha=0.1 (90% coverage)
    with torch.no_grad():
        cp_predictor.calibrate(x_calib.astype(np.float32), y_calib, alpha=0.1)

    # predict on test set
    with torch.no_grad():
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
        torch.manual_seed(seed)
        model = SimpleTorchModel(input_dim=4, n_classes=3)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # convert data to tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        # training loop
        for _ in range(100):
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # disable gradients for evaluation
        model.eval()

        # create predictor with trained model
        score = LACScore(model=model)

        # create conformal predictor
        cp_predictor = SplitConformalClassifier(
            model=model,
            score=score,
            use_accretive=True,
        )

        # calibrate with alpha=0.1
        with torch.no_grad():
            cp_predictor.calibrate(x_calib.astype(np.float32), y_calib, alpha=0.1)

        # predict on test set
        with torch.no_grad():
            prediction_sets = cp_predictor.predict(x_test.astype(np.float32), alpha=0.1)

        # compute empirical coverage
        coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(len(y_test))])
        coverages.append(coverage)

        # coverage should be at least 0.85
        assert coverage >= 0.85, f"coverage {coverage} is below expected 0.85 for seed {seed}"

    # verify coverage is non-empty across all seeds
    assert len(coverages) == 3
    assert all(c >= 0.85 for c in coverages)
