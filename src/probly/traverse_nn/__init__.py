"""Traverser utilities for neural networks."""

from . import nn, torch

## NN

LAYER_COUNT = nn.LAYER_COUNT
is_first_layer = nn.is_first_layer

layer_count_traverser = nn.layer_count_traverser
nn_traverser = nn.nn_traverser

nn_compose = nn.compose

## Torch

TORCH_CLONE = torch.CLONE
TORCH_FLATTEN_SEQUENTIAL = torch.FLATTEN_SEQUENTIAL

torch_traverser = torch._torch_traverser  # noqa: SLF001
