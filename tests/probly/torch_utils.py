"""Util functions for torch tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


def count_layers(model: nn.Module, layer_type: type[nn.Module]) -> int:
    """Counts the number of layers of a specific type in a neural network model.

    This function iterates through all the modules in the given model and counts
    how many of them match the specified layer type. It's particularly useful
    for analyzing the architecture of a neural network or verifying its
    composition.

    Parameters:
        model: The neural network model containing the layers to be counted.
        layer_type: The type of layer to count within the model.

    Returns:
        The number of layers of the specified type found in the model.
    """
    return sum(1 for m in model.modules() if isinstance(m, layer_type))
