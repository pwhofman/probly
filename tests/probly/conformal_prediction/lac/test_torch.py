from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim

from probly.conformal_prediction.lac.torch import LAC

# Constants for deterministic testing
PROB_TRUE_CLASS = 0.8
EXPECTED_SCORE = 1.0 - PROB_TRUE_CLASS  # Expected non-conformity score: 0.2


class MockTorchModel(nn.Module):
    """Mock PyTorch model returning deterministic probabilities for testing."""

    def __init__(self, n_classes: int = 3, true_prob: float = 0.9) -> None:
        """Initialize the mock model."""
        super().__init__()
        self.n_classes = n_classes
        self.true_prob = true_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        device = x.device

        probs = torch.zeros(self.n_classes, device=device)
        probs[0] = self.true_prob

        if self.n_classes > 1:
            remaining = (1.0 - self.true_prob) / (self.n_classes - 1)
            probs[1:] = remaining

        return probs.repeat(n_samples, 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        # Preserve device placement
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")

        # Create probabilities
        probs = torch.zeros(self.n_classes, device=device)

        # Set the true class probability explicitly
        # We use 0.9 to align with the observed threshold of 0.1 (1.0 - 0.9)
        probs[0] = self.true_prob

        # Distribute remaining probability
        if self.n_classes > 1:
            remaining = (1.0 - self.true_prob) / (self.n_classes - 1)
            probs[1:] = remaining

        return probs.repeat(n_samples, 1)


def test_torch_lac_basic_flow() -> None:
    """Verify complete calibration and prediction workflow with Tensor inputs."""
    # We set probability to 0.9. Expected score = 1.0 - 0.9 = 0.1.
    prob_true = 0.9
    expected_score = 1.0 - prob_true  # 0.1

    model = MockTorchModel(true_prob=prob_true)
    predictor = LAC(model)

    # 1. Calibration Data
    # Linter prefers lowercase variables in functions (x_cal instead of X_cal)
    x_cal = torch.randn(10, 5)
    y_cal = torch.zeros(10, dtype=torch.long)  # Class 0 is true

    # 2. Calibrate
    threshold = predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    assert predictor.is_calibrated
    assert threshold is not None

    # Check if threshold matches expected score
    assert abs(threshold - expected_score) < 1e-4, f"Threshold mismatch! Expected {expected_score}, got {threshold}"

    # 3. Predict
    x_test = torch.randn(5, 5)
    sets = predictor.predict(x_test, significance_level=0.1)

    # 4. Verify Output Types
    assert isinstance(sets, torch.Tensor), "Output must be a PyTorch Tensor"
    assert sets.dtype == torch.bool, "Output must be a boolean Tensor"
    assert sets.shape == (5, 3)

    # Class 0 must be included (Score 0.1 <= Threshold 0.1)
    assert torch.all(sets[:, 0]), "Class 0 should be included in prediction sets"


def test_torch_device_preservation() -> None:
    """Ensure that input device (CPU/GPU) is preserved in the output."""
    device = torch.device("cpu")

    model = MockTorchModel()
    predictor = LAC(model)

    x_cal = torch.randn(10, 5).to(device)
    y_cal = torch.zeros(10, dtype=torch.long).to(device)

    predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    x_test = torch.randn(2, 5).to(device)
    sets = predictor.predict(x_test, significance_level=0.1)

    assert sets.device.type == device.type, "Output tensor is on wrong device"


def test_numpy_input_compatibility() -> None:
    """Verify that the wrapper accepts Numpy inputs and returns Tensors."""
    model = MockTorchModel()
    predictor = LAC(model)

    # Use modern numpy random generator to satisfy NPY002
    rng = np.random.default_rng(42)
    x_cal = rng.standard_normal((10, 5)).astype(np.float32)
    y_cal = np.zeros(10, dtype=int)

    predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    x_test = rng.standard_normal((2, 5)).astype(np.float32)
    sets = predictor.predict(x_test, significance_level=0.1)

    assert isinstance(sets, torch.Tensor)


def test_torch_randomized_stress_check() -> None:
    """Stress test with random data to verify robustness and admissibility.

    Ensures 'Accretive Completion' prevents empty sets even with random noise.
    """
    n_samples = 100
    n_classes = 10
    n_features = 5

    x_cal = torch.randn(n_samples, n_features)
    y_cal = torch.randint(0, n_classes, (n_samples,))
    x_test = torch.randn(n_samples, n_features)

    model = MockTorchModel(n_classes=n_classes)

    predictor = LAC(model)
    predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    sets = predictor.predict(x_test, significance_level=0.1)

    assert sets.shape == (n_samples, n_classes)

    # Admissibility Check: Ensure no empty sets (sum of True per row >= 1)
    set_sizes = sets.sum(dim=1)
    assert not torch.any(set_sizes == 0), "Found empty sets (Accretive Completion failure)"


def test_iris_coverage_integration() -> None:
    """Integration Test on Iris Dataset.

    Trains a real (simple) PyTorch model on Iris, calibrates LAC,
    and checks if the empirical coverage matches the target significance level.
    """
    # 1. Prepare Data
    iris = load_iris()
    x_raw = iris.data
    y_raw = iris.target

    # Standardize features (important for Neural Nets)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)

    # Convert to Tensors
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)

    # Split: Train (40%), Calibrate (40%), Test (20%)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_tensor,
        y_tensor,
        test_size=0.6,
        random_state=42,
    )
    x_cal, x_test, y_cal, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.33,
        random_state=42,
    )

    # 2. Define and Train a Wrapper Model
    class IrisModelWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(4, 16)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(16, 3)
            self.softmax = nn.Softmax(dim=1)  # LAC expects probabilities

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return self.softmax(x)

        def predict(self, x: torch.Tensor) -> torch.Tensor:
            """Predict method for compatibility with older tests."""
            self.eval()
            with torch.no_grad():
                return self.forward(x)

    model = IrisModelWrapper()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Quick training loop
    torch.manual_seed(42)
    for _ in range(200):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # 3. Apply LAC
    lac_predictor = LAC(model)

    # Calibrate (Target: 90% coverage => alpha=0.1)
    target_alpha = 0.1
    lac_predictor.calibrate(x_cal, y_cal, significance_level=target_alpha)

    assert lac_predictor.is_calibrated

    # Predict on Test Set
    prediction_sets = lac_predictor.predict(x_test, significance_level=target_alpha)

    # 4. Verify Coverage
    covered = prediction_sets.gather(1, y_test.unsqueeze(1)).squeeze()
    empirical_coverage = covered.float().mean().item()

    # Expectation: Coverage should be roughly 1 - alpha (0.9)
    assert 0.8 <= empirical_coverage <= 1.0, f"Coverage on Iris too low! Expected ~0.9, got {empirical_coverage:.2f}"

    # 5. Admissibility Check (Real Data)
    set_sizes = prediction_sets.sum(dim=1)
    assert not torch.any(set_sizes == 0), "Real data produced empty sets!"


def test_iris_accretive_completion_active() -> None:
    """Integration Test: Forced Accretive Completion on Iris.

    Simulates a scenario where standard calibration produces empty sets
    (by forcing an impossibly strict threshold) to verify that the
    'Accretive Completion' fallback logic correctly activates and
    rescues the top-1 prediction.
    """
    # 1. Prepare Data
    iris = load_iris()
    x_tensor = torch.tensor(StandardScaler().fit_transform(iris.data), dtype=torch.float32)
    y_tensor = torch.tensor(iris.target, dtype=torch.long)

    # Use stratification to ensure consistent class balance in small splits
    x_train, x_test, y_train, y_test = train_test_split(
        x_tensor,
        y_tensor,
        test_size=0.5,
        random_state=42,
        stratify=y_tensor,
    )

    # 2. Define and Train Model
    class IrisModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(42)
            self.linear1 = nn.Linear(4, 16)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(16, 3)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return self.softmax(x)

        def predict(self, x: torch.Tensor) -> torch.Tensor:
            """Predict method for compatibility."""
            self.eval()
            with torch.no_grad():
                return self.forward(x)

    model = IrisModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train for 50 epochs to get meaningful probabilities
    for _ in range(50):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # 3. Initialize LAC
    predictor = LAC(model)
    # Perform standard calibration to set internal state (is_calibrated=True)
    predictor.calibrate(x_train, y_train, significance_level=0.1)

    # 4. Force "Impossible" Threshold
    # LAC inclusion condition: score <= threshold <=> (1 - p) <= t <=> p >= 1 - t
    # We enforce threshold = 0.0.
    # New condition: p >= 1.0 - 0.0 => p >= 1.0
    # Since Softmax outputs are rarely exactly 1.0, this rejects all classes initially,
    # creating empty prediction sets (Null Regions).
    predictor.threshold = 0.0

    # 5. Predict (Triggering Accretive Completion)
    # The empty sets must be repaired by adding the class with the highest probability.
    sets = predictor.predict(x_test, significance_level=0.1)

    # 6. Verify Results
    # a) Admissibility check: No empty sets allowed
    set_sizes = sets.sum(dim=1)
    assert not torch.any(set_sizes == 0), "Accretive completion failed to fix empty sets!"

    # b) Check Set Size
    # Since we raised the bar to 1.0, only the 'rescue' mechanism adds classes.
    # It adds classes until the set is non-empty (typically just the top-1).
    avg_size = set_sizes.float().mean().item()

    assert avg_size == 1.0, f"Expected exactly 1 class per sample (Top-1 rescue), got avg {avg_size:.2f}"
