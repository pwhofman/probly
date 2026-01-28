"""Tests for PyTorch CQR implementation (inspired by APS)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Type Checking Support ---
if TYPE_CHECKING:
    import torch
    from torch import nn
    import torch.nn.functional as F

# --- Runtime Support ---
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

# Import our CQR modules
from probly.conformal_prediction.scores.cqr.common import CQRScore, cqr_score_func
from probly.conformal_prediction.scores.cqr.torch import cqr_score_torch

# --- 1. Helper Classes (Model & Loss) ---


class SimpleQuantileNet(nn.Module):
    """A simple neural network for quantile regression testing.

    Outputs 2 values: [lower_quantile, upper_quantile]
    """

    def __init__(self, input_dim: int) -> None:
        """Initialize the quantile network."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)  # Output: q_lo, q_hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        x = F.relu(self.fc1(x))
        # Mypy treats torch.Tensor as Any due to conditional import, so we ignore the warning
        return cast(torch.Tensor, self.fc2(x))


class TorchModelWrapper:
    """Wraps a PyTorch model to handle numpy inputs automatically.

    This acts as the 'Predictor' for CQRScore, ensuring the PyTorch model
    receives Tensors even if CQRScore passes numpy arrays.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the wrapper with a PyTorch model."""
        self.model = model

    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic numpy-to-tensor conversion."""
        # 1. Convert numpy to tensor if needed
        if not isinstance(x, torch.Tensor):
            # Using float32 is standard for Torch
            x = torch.tensor(x, dtype=torch.float32)

        # 2. Forward pass
        # We detach() so it can be converted back to numpy by CQRScore later if needed
        # Mypy treats torch.Tensor as Any due to conditional import, so we ignore the warning
        return cast(torch.Tensor, self.model(x).detach())


def quantile_loss(preds: torch.Tensor, target: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Helper: Pinball loss to train the quantile regressor."""
    target = target.view(-1, 1)

    # Lower quantile (alpha/2)
    q_lo = preds[:, 0:1]
    err_lo = target - q_lo
    loss_lo = torch.max((alpha / 2) * err_lo, ((alpha / 2) - 1) * err_lo)

    # Upper quantile (1 - alpha/2)
    q_hi = preds[:, 1:2]
    err_hi = target - q_hi
    loss_hi = torch.max((1 - alpha / 2) * err_hi, ((1 - alpha / 2) - 1) * err_hi)

    return torch.mean(loss_lo + loss_hi)


# --- 2. Fixtures ---


@pytest.fixture
def simple_model() -> nn.Module:
    """Fixture to create a simple PyTorch quantile model."""
    return SimpleQuantileNet(input_dim=10)  # Diabetes dataset has 10 features


# --- 3. Test Class ---


class TestCQRScoreTorch:
    """Test class for CQRScore using PyTorch models."""

    def test_cqr_score_torch_gradients(self) -> None:
        """Test that gradients flow through the score calculation (Backward Pass)."""
        # Create inputs that require gradients
        y_pred = torch.tensor([[4.0, 6.0]], requires_grad=True)
        y_true = torch.tensor([2.0])  # Value is 2.0 (Below lower bound 4.0)

        # Forward Pass via direct function call
        score = cqr_score_torch(y_true, y_pred)

        # Check value (4.0 - 2.0 = 2.0)
        assert torch.isclose(score[0], torch.tensor(2.0))

        # Backward Pass
        score.backward()

        # Gradient check: d(Score)/d(lower) should be 1.0
        assert y_pred.grad is not None
        assert torch.isclose(y_pred.grad[0, 0], torch.tensor(1.0))

    def test_cqrscore_with_torch_model(self, simple_model: nn.Module) -> None:
        """Test CQRScore wrapper with a PyTorch model."""
        # Wrap the model so it accepts numpy arrays from CQRScore
        wrapped_model = TorchModelWrapper(simple_model)
        score = CQRScore(model=wrapped_model)

        # Create dummy data (numpy)
        rng = np.random.default_rng(42)
        x_calib = rng.random((20, 10)).astype(np.float32)
        y_calib = rng.random((20,)).astype(np.float32)

        # Test calibration scores
        cal_scores = score.calibration_nonconformity(x_calib, y_calib)

        assert isinstance(cal_scores, np.ndarray)
        assert cal_scores.shape == (20,)
        assert np.all(cal_scores >= 0)

    def test_torch_model_forward_pass_shapes(self, simple_model: nn.Module) -> None:
        """Test that PyTorch model forward pass returns correct shapes."""
        rng = np.random.default_rng(42)
        x_test = rng.random((5, 10)).astype(np.float32)

        # convert to tensor
        x_tensor = torch.tensor(x_test)
        preds = simple_model(x_tensor)

        # check shapes: (N, 2)
        assert preds.shape == (5, 2), f"Expected shape (5, 2), got {preds.shape}"

    def test_with_diabetes_dataset(self) -> None:
        """Test with real Diabetes dataset (Training Loop included)."""
        # 1. Load Data
        data = load_diabetes()
        # Rename variables to lowercase to satisfy N806
        x_data = data.data.astype(np.float32)
        y_data = data.target.astype(np.float32)

        # Split: Train (40%), Calib (40%), Test (20%)
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_data,
            y_data,
            test_size=0.6,
            random_state=42,
        )
        x_cal, x_test, y_cal, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.33,
            random_state=42,
        )

        # Scale features
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_cal = scaler.transform(x_cal)
        x_test = scaler.transform(x_test)

        # 2. Train Model
        torch.manual_seed(42)
        model = SimpleQuantileNet(input_dim=x_data.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        x_train_tensor = torch.tensor(x_train)
        y_train_tensor = torch.tensor(y_train)

        model.train()
        for _ in range(200):  # Short training loop
            optimizer.zero_grad()
            preds = model(x_train_tensor)
            loss = quantile_loss(preds, y_train_tensor, alpha=0.1)
            loss.backward()
            optimizer.step()

        model.eval()

        # 3. Apply CQR
        # Wrap the trained model
        score = CQRScore(model=TorchModelWrapper(model))

        # Calculate calibration scores
        cal_scores = score.calibration_nonconformity(x_cal, y_cal)

        # Determine quantile (1 - alpha)
        alpha = 0.1
        n = len(y_cal)
        q_val = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")

        # 4. Validate on Test Set
        # Predict intervals (widths)
        widths = score.predict_nonconformity(x_test)  # returns q_hi - q_lo
        # Check widths shape to satisfy F841 (variable usage)
        assert widths.shape == (len(x_test), 1)

        # Manually verify coverage
        with torch.no_grad():
            raw_preds = model(torch.tensor(x_test))
            q_lo = raw_preds[:, 0].numpy()
            q_hi = raw_preds[:, 1].numpy()

        lower_bound = q_lo - q_val
        upper_bound = q_hi + q_val

        # Check coverage
        covered = (y_test >= lower_bound) & (y_test <= upper_bound)
        coverage = np.mean(covered)

        assert coverage >= 0.8, f"Coverage {coverage} is too low!"

    def test_dispatch_works_with_tensors(self) -> None:
        """Test that calling generic cqr_score_func with Tensors works."""
        y_true = torch.tensor([5.0])
        y_pred = torch.tensor([[4.0, 6.0]])

        # This calls the lazy dispatch mechanism
        scores = cqr_score_func(y_true, y_pred)

        assert isinstance(scores, torch.Tensor)
        assert scores.item() == 0.0

    def test_cqr_score_torch_crossing_quantiles(self) -> None:
        """Test behavior when model predicts crossing quantiles (q_lo > q_hi).

        This often happens early in training. CQR should handle this gracefully
        and return a high nonconformity score (penalizing the invalid interval).
        """
        # Target y is 0
        y_true = torch.tensor([0.0])

        # Model predicts: q_lo = 2.0, q_hi = -2.0 (Crossed! Lower is higher than Upper)
        y_pred = torch.tensor([[2.0, -2.0]])

        # Fix Mypy error by adding explicit type hint
        scores: torch.Tensor = cqr_score_torch(y_true, y_pred)

        # Logic check:
        # diff_lower = q_lo - y = 2.0 - 0.0 = 2.0
        # diff_upper = y - q_hi = 0.0 - (-2.0) = 2.0
        # max(2.0, 2.0, 0) = 2.0

        assert scores.item() == 2.0
        assert scores.item() >= 0  # Score must never be negative
