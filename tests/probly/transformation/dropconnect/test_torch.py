"""Test for torch dropconnect models."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.transformation.dropconnect import dropconnect


@pytest.mark.parametrize("p_value", [0.0, 0.25, 0.5, 0.75, 0.99])
def test_dropconnect_different_probabilities(torch_model_small_2d_2d: nn.Module, p_value: float) -> None:
    """Test dropconnect with different probability values using fixtures."""
    transformed_model = dropconnect(torch_model_small_2d_2d, p=p_value)

    # Test forward pass
    x = torch.randn(4, 2)
    output = transformed_model(x)

    # Basic validation
    assert output is not None
    assert output.shape == (4, 2)


def test_dropconnect_replaces_linear_layers(torch_model_small_2d_2d: nn.Module) -> None:
    """Test that dropconnect replaces Linear layers with DropConnect layers."""
    transformed_model = dropconnect(torch_model_small_2d_2d, p=0.5)

    # Check that at least one layer was replaced
    linear_layers_replaced = False
    for module in transformed_model.modules():
        if hasattr(module, "__class__") and "DropConnect" in module.__class__.__name__:
            linear_layers_replaced = True
            break

    assert linear_layers_replaced, "No Linear layers were replaced with DropConnect layers"


def test_dropconnect_complex_model(torch_custom_model: nn.Module) -> None:
    """Test dropconnect on more complex model fixtures."""
    transformed_model = dropconnect(torch_custom_model, p=0.3)

    x = torch.randn(2, 10)
    output = transformed_model(x)

    assert output is not None
