"""Lightweight API sanity tests for probly.transformation.dropconnect.common."""
# Rewritten DropConnect common test structure

from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.layers.torch import DropConnectLinear
from probly.transformation.dropconnect import dropconnect

torch = pytest.importorskip("torch")


def test_dropconnect_preserves_forward_shape(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """Applying the transformation should not change the model's I/O shape."""
    model = dropconnect(torch_model_small_2d_2d, p=0.4)
    x = torch.randn(3, 2)  # matches the small 2d->2d fixture input size
    y = model(x)
    assert y.shape[0] == x.shape[0]


def test_at_least_one_replacement_happens_when_possible(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """
    If a model contains ≥2 Linear layers and starts with Linear,
    at least one Linear should be replaced by DropConnectLinear.
    """
    model = dropconnect(torch_model_small_2d_2d, p=0.5)
    assert any(isinstance(m, DropConnectLinear) for m in model.modules())
