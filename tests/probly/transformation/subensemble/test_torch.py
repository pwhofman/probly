from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.transformation.ensemble.torch import generate_torch_ensemble


class DummyNet(nn.Module):
    """A small MLP used for testing ensemble behavior."""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward pass."""
        return self.fc2(self.fc1(x))


def test_ensemble_returns_modulelist() -> None:
    """The factory should return nn.ModuleList."""
    model = DummyNet()
    ensemble = generate_torch_ensemble(model, num_members=4)

    assert isinstance(ensemble, nn.ModuleList)
    assert len(ensemble) == 4


def test_members_are_deep_copied() -> None:
    """Members must be deep-copied, not shared."""
    model = DummyNet()
    ensemble = generate_torch_ensemble(model, num_members=3, reset_params=False)

    p0 = next(iter(ensemble[0].parameters())).detach().clone()
    p1 = next(iter(ensemble[1].parameters())).detach().clone()

    assert p0.data_ptr() != p1.data_ptr()


def test_reset_params_changes_initialization() -> None:
    """Resetting parameters should result in different initialization."""
    model = DummyNet()
    ensemble = generate_torch_ensemble(model, num_members=2, reset_params=True)

    params0 = next(iter(ensemble[0].parameters())).detach().clone()
    params1 = next(iter(ensemble[1].parameters())).detach().clone()

    assert not torch.equal(params0, params1)


def test_no_reset_keeps_same_initialization() -> None:
    """Without reset, members should have identical parameters."""
    model = DummyNet()
    ensemble = generate_torch_ensemble(model, num_members=2, reset_params=False)

    params0 = next(iter(ensemble[0].parameters())).detach().clone()
    params1 = next(iter(ensemble[1].parameters())).detach().clone()

    assert torch.equal(params0, params1)


def test_forward_batch_compatibility() -> None:
    """Each copied model should process a batch normally."""
    model = DummyNet()
    ensemble = generate_torch_ensemble(model, num_members=3)

    x = torch.randn(4, 10)

    for member in ensemble:
        out = member(x)
        assert out.shape == (4, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_ensemble_gpu_cpu_compatibility() -> None:
    """Models should run correctly on GPU if available."""
    model = DummyNet().cuda()
    ensemble = generate_torch_ensemble(model, num_members=2)

    x = torch.randn(3, 10).cuda()

    for member in ensemble:
        out = member(x)
        assert out.is_cuda
        assert out.shape == (3, 2)


def test_empty_ensemble() -> None:
    """num_members = 0 should return empty ModuleList."""
    model = DummyNet()
    ensemble = generate_torch_ensemble(model, num_members=0)

    assert isinstance(ensemble, nn.ModuleList)
    assert len(ensemble) == 0


def test_single_member_ensemble() -> None:
    """num_members = 1 should produce a single model copy."""
    model = DummyNet()
    ensemble = generate_torch_ensemble(model, num_members=1)

    assert len(ensemble) == 1
    assert isinstance(ensemble[0], nn.Module)
