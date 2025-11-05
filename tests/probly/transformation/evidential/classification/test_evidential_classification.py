"""Test package for evidential transformations."""

from __future__ import annotations

import torch
from torch import nn

from probly.transformation.evidential.classification.torch import append_activation_torch


def test_returns_sequential_and_appends_softplus() -> None:
    """Ensure Softplus is appended to a Sequential model."""
    base = nn.Linear(4, 2)
    model = append_activation_torch(base)

    assert isinstance(model, nn.Sequential)
    assert model[0] is base
    assert isinstance(model[1], nn.Softplus)

    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 2)
    assert torch.isfinite(y).all()


def test_softplus_effect_on_output() -> None:
    """Check that Softplus produces strictly positive outputs."""
    base = nn.Identity()
    model = append_activation_torch(base)

    x = torch.tensor([[-2.0, 0.0, 1.0]])
    y = model(x)

    assert y.shape == x.shape
    assert torch.all(y > 0)


def test_append_no_param_change() -> None:
    """Verify appending Softplus does not alter parameters."""
    base = nn.Linear(3, 3)
    before = [p.detach().clone() for p in base.parameters()]

    model = append_activation_torch(base)
    assert model[0] is base

    after = list(base.parameters())
    for p_before, p_after in zip(before, after, strict=False):
        assert torch.equal(p_before, p_after)
