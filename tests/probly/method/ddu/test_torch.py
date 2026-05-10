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


# --- Lower-level traverser & helper coverage ---


def _torch_nn():
    """Return torch + nn modules or skip the calling test if torch missing."""
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415
    from torch import nn as _nn  # noqa: PLC0415

    return _torch, _nn


class TestDDURepresentation:
    """The TorchDDURepresentation dataclass."""

    def test_can_construct(self) -> None:
        _torch, _ = _torch_nn()
        from probly.method.ddu.torch import TorchDDURepresentation  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        rep = TorchDDURepresentation(
            softmax=TorchProbabilityCategoricalDistribution(_torch.tensor([[0.5, 0.5]])),
            densities=_torch.tensor([[1.0, 2.0]]),
        )
        assert rep.densities.shape == (1, 2)


class TestNegativeLogDensity:
    """Negative log density convertor."""

    def test_with_torch_tensor(self) -> None:
        _torch, _ = _torch_nn()
        from probly.method.ddu.torch import torch_negative_log_density  # noqa: PLC0415

        densities = _torch.tensor([[0.0, 0.0], [-1e10, 0.0]])
        out = torch_negative_log_density(densities)
        assert out.shape == (2,)


class TestReLUReplacementTraverser:
    """``torch_ddu_traverser`` replaces ReLU and ReLU6 with LeakyReLU."""

    def test_relu6_replaced_with_leaky_relu(self) -> None:
        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import (  # noqa: PLC0415
            HAS_RESIDUAL,
            HEAD_MODULE,
            SN_COEFF,
            torch_ddu_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        relu6 = _nn.ReLU6()
        out, _ = traverse_with_state(
            relu6,
            nn_compose(torch_ddu_traverser),
            init={HEAD_MODULE: None, SN_COEFF: 3.0, HAS_RESIDUAL: False},
        )
        assert isinstance(out, _nn.LeakyReLU)
        assert out.negative_slope == 0.01

    def test_relu_replaced_with_leaky_relu(self) -> None:
        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import (  # noqa: PLC0415
            HAS_RESIDUAL,
            HEAD_MODULE,
            SN_COEFF,
            torch_ddu_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        out, _ = traverse_with_state(
            _nn.ReLU(),
            nn_compose(torch_ddu_traverser),
            init={HEAD_MODULE: None, SN_COEFF: 3.0, HAS_RESIDUAL: False},
        )
        assert isinstance(out, _nn.LeakyReLU)


class TestTorchDDUPredictorInit:
    """End-to-end DDU predictor construction."""

    def test_no_linear_raises(self) -> None:
        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import TorchDDUPredictor  # noqa: PLC0415

        model = _nn.Sequential(_nn.ReLU(), _nn.Tanh())
        with pytest.raises(ValueError, match=r"No nn\.Linear"):
            TorchDDUPredictor(model)

    def test_no_residual_warns(self) -> None:
        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import TorchDDUPredictor  # noqa: PLC0415

        model = _nn.Sequential(_nn.Linear(4, 8), _nn.ReLU(), _nn.Linear(8, 3))
        with pytest.warns(UserWarning, match="residual"):
            TorchDDUPredictor(model)

    def test_fit_density_head_runs(self) -> None:
        import warnings as _warnings  # noqa: PLC0415

        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import TorchDDUPredictor  # noqa: PLC0415

        _torch.manual_seed(0)
        model = _nn.Sequential(_nn.Linear(4, 8), _nn.ReLU(), _nn.Linear(8, 3))
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            ddu_pred = TorchDDUPredictor(model)
        x = _torch.randn(20, 4)
        labels = _torch.randint(0, 3, (20,))
        ddu_pred.fit_density_head(x, labels)
        assert _torch.any(ddu_pred.density_head.means != 0)

    def test_predict_representation(self) -> None:
        import warnings as _warnings  # noqa: PLC0415

        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import TorchDDUPredictor, TorchDDURepresentation  # noqa: PLC0415

        _torch.manual_seed(0)
        model = _nn.Sequential(_nn.Linear(4, 8), _nn.ReLU(), _nn.Linear(8, 3))
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            ddu_pred = TorchDDUPredictor(model)
        x = _torch.randn(2, 4)
        rep = ddu_pred.predict_representation(x)
        assert isinstance(rep, TorchDDURepresentation)
        assert rep.densities.shape == (2, 3)


class TestResidualDetection:
    """The residual_detection_traverser detects skip connections."""

    def test_detects_torch_add(self) -> None:
        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import (  # noqa: PLC0415
            HAS_RESIDUAL,
            HEAD_MODULE,
            SN_COEFF,
            residual_detection_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        class ResidualBlock(_nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = _nn.Linear(4, 4)

            def forward(self, x):
                return _torch.add(x, self.fc(x))

        block = ResidualBlock()
        _, final_state = traverse_with_state(
            block,
            nn_compose(residual_detection_traverser),
            init={HAS_RESIDUAL: False, HEAD_MODULE: None, SN_COEFF: 3.0},
        )
        assert final_state[HAS_RESIDUAL] is True

    def test_detects_operator_add(self) -> None:
        _torch, _nn = _torch_nn()
        from probly.method.ddu.torch import (  # noqa: PLC0415
            HAS_RESIDUAL,
            HEAD_MODULE,
            SN_COEFF,
            residual_detection_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        class ResidualBlock(_nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = _nn.Linear(4, 4)

            def forward(self, x):
                return x + self.fc(x)

        block = ResidualBlock()
        _, final_state = traverse_with_state(
            block,
            nn_compose(residual_detection_traverser),
            init={HAS_RESIDUAL: False, HEAD_MODULE: None, SN_COEFF: 3.0},
        )
        assert final_state[HAS_RESIDUAL] is True

    def test_returns_unchanged_when_already_detected(self) -> None:
        _, _nn = _torch_nn()
        from probly.method.ddu.torch import (  # noqa: PLC0415
            HAS_RESIDUAL,
            HEAD_MODULE,
            SN_COEFF,
            residual_detection_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        block = _nn.Linear(4, 4)
        _, final_state = traverse_with_state(
            block,
            nn_compose(residual_detection_traverser),
            init={HAS_RESIDUAL: True, HEAD_MODULE: None, SN_COEFF: 3.0},
        )
        assert final_state[HAS_RESIDUAL] is True
