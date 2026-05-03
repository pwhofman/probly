"""Tests for torch DDU implementation."""

from __future__ import annotations

import pytest

from probly.decider import categorical_from_mean
from probly.method.ddu import ddu
from probly.predictor import predict

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """Small MLP with two hidden layers for testing."""

    def __init__(self, in_features: int = 16, hidden: int = 32, num_classes: int = 3) -> None:
        """Initialize SimpleMLP."""
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SimpleMLP."""
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.head(x)


class DownsampleNet(nn.Module):
    """Small CNN with a stride-1x1 downsampling convolution (residual branch pattern)."""

    def __init__(self, num_classes: int = 3) -> None:
        """Initialize DownsampleNet."""
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv2d(16, 32, kernel_size=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for DownsampleNet."""
        x = self.relu(self.conv(x))
        x = self.downsample(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
IN_FEATURES = 16
HIDDEN = 32


@pytest.fixture
def mlp_model() -> SimpleMLP:
    torch.manual_seed(42)
    return SimpleMLP(in_features=IN_FEATURES, hidden=HIDDEN, num_classes=NUM_CLASSES)


@pytest.fixture
def downsample_model() -> DownsampleNet:
    torch.manual_seed(42)
    return DownsampleNet(num_classes=NUM_CLASSES)


@pytest.fixture
def test_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, IN_FEATURES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpectralNorm:
    """Test that spectral normalization is applied to the correct layers."""

    def test_sn_applied_to_hidden_linear(self, mlp_model: SimpleMLP) -> None:
        """Hidden Linear layers should receive spectral normalisation."""
        out = ddu(mlp_model)
        sn_count = sum(1 for m in out.modules() if hasattr(m, "parametrizations") and "weight" in m.parametrizations)
        assert sn_count >= 2  # fc1 and fc2, not head

    def test_head_not_parametrized(self, mlp_model: SimpleMLP) -> None:
        """The last Linear layer (head) must not receive spectral normalisation."""
        out = ddu(mlp_model)
        linears = [m for m in out.modules() if isinstance(m, nn.Linear)]
        head = linears[-1]
        assert not (hasattr(head, "parametrizations") and "weight" in head.parametrizations)

    def test_sn_coeff_stored(self, mlp_model: SimpleMLP) -> None:
        """The SN parametrization should use the provided coefficient."""
        out = ddu(mlp_model, sn_coeff=5.0)
        params = [
            p
            for m in out.modules()
            if hasattr(m, "parametrizations") and "weight" in m.parametrizations
            for p in m.parametrizations.weight
        ]
        assert all(p.coeff == 5.0 for p in params)


class TestReLUReplacement:
    """Test that ReLU and ReLU6 are replaced with LeakyReLU."""

    def test_no_relu_remains(self, mlp_model: SimpleMLP) -> None:
        """No nn.ReLU or nn.ReLU6 modules should remain after transformation."""
        out = ddu(mlp_model)
        assert not any(isinstance(m, (nn.ReLU, nn.ReLU6)) for m in out.modules())

    def test_leaky_relu_added(self, mlp_model: SimpleMLP) -> None:
        """LeakyReLU modules should be present after transformation."""
        out = ddu(mlp_model)
        assert any(isinstance(m, nn.LeakyReLU) for m in out.modules())

    def test_leaky_relu_slope(self, mlp_model: SimpleMLP) -> None:
        """Replaced activations should use negative_slope=0.01."""
        out = ddu(mlp_model)
        for m in out.modules():
            if isinstance(m, nn.LeakyReLU):
                assert m.negative_slope == pytest.approx(0.01)


class TestDownsamplingFix:
    """Test that stride-1x1 downsampling convolutions are replaced."""

    def test_no_stride_1x1_conv_remains(self, downsample_model: DownsampleNet) -> None:
        """Conv2d(kernel_size=1, stride>1) should be replaced after transformation."""
        out = ddu(downsample_model)
        bad = [
            m
            for m in out.modules()
            if isinstance(m, nn.Conv2d)
            and (m.stride[0] if isinstance(m.stride, (tuple, list)) else m.stride) > 1
            and (m.kernel_size[0] if isinstance(m.kernel_size, (tuple, list)) else m.kernel_size) == 1
        ]
        assert len(bad) == 0

    def test_avgpool_inserted(self, downsample_model: DownsampleNet) -> None:
        """An AvgPool2d should be inserted where a downsampling conv was removed."""
        out = ddu(downsample_model)
        assert any(isinstance(m, nn.AvgPool2d) for m in out.modules())


class TestForwardPass:
    """Test that the transformed model still produces valid outputs."""

    def test_output_shape_mlp(self, mlp_model: SimpleMLP, test_input: torch.Tensor) -> None:
        """Transformed MLP should produce logits of correct shape."""
        out = ddu(mlp_model)
        logits, densities = out(test_input)
        assert logits.shape == (test_input.shape[0], NUM_CLASSES)
        assert densities.shape == (test_input.shape[0], NUM_CLASSES)

    def test_categorical_from_mean_returns_ddu_softmax(self, mlp_model: SimpleMLP, test_input: torch.Tensor) -> None:
        """The categorical mean decider should reduce DDU representations to their softmax distribution."""
        out = ddu(mlp_model)

        single = categorical_from_mean(predict(out, test_input))
        logits, _ = out(test_input)

        assert isinstance(single, TorchCategoricalDistribution)
        assert torch.allclose(single.probabilities, torch.softmax(logits, dim=-1))

    def test_output_shape_downsample(self, downsample_model: DownsampleNet) -> None:
        """Transformed DownsampleNet should produce logits of correct shape."""
        out = ddu(downsample_model)
        x = torch.randn(2, 3, 8, 8)
        logits, densities = out(x)
        assert logits.shape == (2, NUM_CLASSES)
        assert densities.shape == (2, NUM_CLASSES)
