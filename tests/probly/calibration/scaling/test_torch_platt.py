from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from probly.calibration.scaling.torch_platt import TorchPlatt

ReturnTypeModelFixture = tuple[TorchPlatt, nn.Module]
ReturnTypeLoaderFixture = tuple[DataLoader, Tensor]


@pytest.fixture
def setup_model(torch_binary_model: nn.Sequential) -> ReturnTypeModelFixture:
    """Set up a TorchAffine (Platt scaling) model."""
    base = torch_binary_model
    affine_model = TorchPlatt(base)
    return affine_model, base


@pytest.fixture
def setup_calibration_loader() -> ReturnTypeLoaderFixture:
    """Set up a small binary dataset and loader."""
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    return loader, x


def test_forward(setup_model: ReturnTypeModelFixture, setup_calibration_loader: ReturnTypeLoaderFixture) -> None:
    """Check if forward returns scaled logits."""
    affine_model, base = setup_model
    _, x = setup_calibration_loader

    logits = base(x)
    scaled_logits = affine_model(x)

    expected = logits * affine_model.w + affine_model.b
    assert torch.allclose(scaled_logits, expected, atol=1e-5)


def test_fit_updates_parameters(
    setup_model: ReturnTypeModelFixture,
    setup_calibration_loader: ReturnTypeLoaderFixture,
) -> None:
    """Check if fit updates Platt scaling parameters."""
    affine_model, _ = setup_model
    loader, _ = setup_calibration_loader

    # Save initial parameters
    initial_w = affine_model.w.clone()
    initial_b = affine_model.b.clone()

    affine_model.fit(loader, learning_rate=0.1, max_iter=10)

    # Ensure parameters changed
    assert not torch.allclose(affine_model.w, initial_w)
    assert not torch.allclose(affine_model.b, initial_b)


def test_predict_returns_probabilities(
    setup_model: ReturnTypeModelFixture,
    setup_calibration_loader: ReturnTypeLoaderFixture,
) -> None:
    """Check that predict returns valid probabilities between 0 and 1."""
    affine_model, _ = setup_model
    _, x = setup_calibration_loader

    probs = affine_model.predict(x)

    # Binary classification → shape (N, 1)
    assert probs.shape == (20, 1)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
