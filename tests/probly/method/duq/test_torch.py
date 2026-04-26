"""Tests for the torch DUQ implementation."""

from __future__ import annotations

import pytest

from probly.method.duq import duq
from probly.predictor import predict
from probly.quantification import quantify

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

NUM_CLASSES = 4
IN_FEATURES = 8
HIDDEN = 16
CENTROID_SIZE = 32


class SimpleMLP(nn.Module):
    """Two-layer MLP used as a test base model."""

    def __init__(self) -> None:
        """Initialize SimpleMLP."""
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.relu = nn.ReLU()
        self.head = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SimpleMLP."""
        return self.head(self.relu(self.fc1(x)))


@pytest.fixture
def mlp() -> SimpleMLP:
    torch.manual_seed(0)
    return SimpleMLP()


@pytest.fixture
def batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(5, IN_FEATURES)


class TestArchitecture:
    """The transformation should swap the head and attach a centroid head."""

    def test_head_replaced_with_identity(self, mlp: SimpleMLP) -> None:
        out = duq(mlp, centroid_size=CENTROID_SIZE)
        # Encoder no longer contains a Linear-shaped classification head.
        linears_in_encoder = [m for m in out.encoder.modules() if isinstance(m, nn.Linear)]
        # Only the hidden Linear should remain; the head was replaced by Identity.
        assert len(linears_in_encoder) == 1
        assert linears_in_encoder[0].out_features == HIDDEN

    def test_centroid_head_dimensions(self, mlp: SimpleMLP) -> None:
        out = duq(mlp, centroid_size=CENTROID_SIZE)
        head = out.centroid_head
        assert head.feature_dim == HIDDEN
        assert head.num_classes == NUM_CLASSES
        assert head.weight.shape == (CENTROID_SIZE, NUM_CLASSES, HIDDEN)
        assert head.centroids_sum.shape == (CENTROID_SIZE, NUM_CLASSES)
        assert head.centroid_counts.shape == (NUM_CLASSES,)


class TestForward:
    """The forward pass should produce per-class kernel values in [0, 1]."""

    def test_kernel_shape(self, mlp: SimpleMLP, batch: torch.Tensor) -> None:
        out = duq(mlp, centroid_size=CENTROID_SIZE)
        k = out(batch)
        assert k.shape == (batch.shape[0], NUM_CLASSES)

    def test_kernel_in_unit_interval(self, mlp: SimpleMLP, batch: torch.Tensor) -> None:
        out = duq(mlp, centroid_size=CENTROID_SIZE)
        k = out(batch)
        assert torch.all(k >= 0.0)
        assert torch.all(k <= 1.0)


class TestCentroidUpdate:
    """The EMA update should change the stored centroid statistics."""

    def test_update_changes_counts(self, mlp: SimpleMLP, batch: torch.Tensor) -> None:
        out = duq(mlp, centroid_size=CENTROID_SIZE, gamma=0.5)
        before = out.centroid_head.centroid_counts.clone()
        labels = torch.arange(batch.shape[0]) % NUM_CLASSES
        out.update_centroids(batch, nn.functional.one_hot(labels, NUM_CLASSES).float())
        after = out.centroid_head.centroid_counts
        assert not torch.allclose(before, after)


class TestRepresentationAndQuantification:
    """predict + quantify should return a per-sample uncertainty score."""

    def test_quantify_shape(self, mlp: SimpleMLP, batch: torch.Tensor) -> None:
        out = duq(mlp, centroid_size=CENTROID_SIZE)
        rep = predict(out, batch)
        unc = quantify(rep)
        assert unc.shape == (batch.shape[0],)
        assert torch.all(unc >= 0.0)
        assert torch.all(unc <= 1.0)
