"""Tests for the dropout module."""

from __future__ import annotations

import pytest
import torch

from probly.representation.dropout import Dropout


@pytest.mark.parametrize("probability", [0.1, 0.5])
def test_dropout(
    model_small_2d_2d: torch.nn.Module,
    probability: float,
) -> None:
    """Test for the dropout module.

    Args:
        model_small_2d_2d: A small model with 2 input and 2 output neurons.
        probability: The probability of dropping out a neuron.

    """
    dropout_model = Dropout(model_small_2d_2d, p=probability)
    assert dropout_model.p == probability
    assert dropout_model.model is not None
    list_of_layers = list(dropout_model.model.children())
    for layer in list_of_layers:
        for _, module in layer.named_modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.p == probability
