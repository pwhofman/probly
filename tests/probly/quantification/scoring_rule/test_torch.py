"""Tests for proper scoring rule loss vectors on PyTorch tensors."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification.scoring_rule import BrierLoss, LogLoss, SphericalLoss, ZeroOneLoss


def test_torch_log_loss_vector() -> None:
    p = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float64)
    assert torch.allclose(LogLoss().loss(p), -torch.log(p), rtol=1e-12, atol=1e-12)


def test_torch_brier_loss_vector() -> None:
    p = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(BrierLoss().loss(p), torch.tensor([[0.0, 2.0]], dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_torch_zero_one_loss_vector() -> None:
    p = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float64)
    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(ZeroOneLoss().loss(p), expected, rtol=1e-12, atol=1e-12)


def test_torch_spherical_loss_vector() -> None:
    p = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(
        SphericalLoss().loss(p), torch.tensor([[0.0, 1.0]], dtype=torch.float64), rtol=1e-12, atol=1e-12
    )


def test_torch_loss_preserves_shape() -> None:
    p = torch.full((4, 3, 5), 1.0 / 5.0, dtype=torch.float64)
    for rule in (LogLoss(), BrierLoss(), ZeroOneLoss(), SphericalLoss()):
        assert rule.loss(p).shape == p.shape
