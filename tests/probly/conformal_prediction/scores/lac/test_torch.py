"""Tests for PyTorch LAC implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from probly.conformal_prediction.methods.common import predict_probs
from probly.conformal_prediction.methods.split import SplitConformalPredictor
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


@predict_probs.register(MockTorchModel)
def predict_probs_mock_torch(model: MockTorchModel, x: Sequence[Any]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return model.forward(x)
    return model.forward(torch.tensor(x, dtype=torch.float32))


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
    """Test LACScore integrated in SplitConformalPredictor."""
    model = MockTorchModel(true_prob=0.8)
    score = LACScore(model=model)
    predictor = SplitConformalPredictor(model=model, score=score, use_accretive=True)

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


def test_lacscore_with_accretive_completion() -> None:
    """Test LACScore with accretive completion."""
    # Create a model that gives very low probabilities
    # This might cause empty prediction sets without accretive completion
    model = MockTorchModel(true_prob=0.1)  # Very low probability

    # Test WITHOUT accretive completion
    score = LACScore(model=model)
    _predictor_no_accretive = SplitConformalPredictor(
        model=model,
        score=score,
        use_accretive=False,
    )

    # Test WITH accretive completion
    _predictor_with_accretive = SplitConformalPredictor(
        model=model,
        score=score,
        use_accretive=True,
    )
