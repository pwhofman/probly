"""Traverser utilities for neural networks."""

from . import nn

## NN

LAYER_COUNT = nn.LAYER_COUNT
is_first_layer = nn.is_first_layer

layer_count_traverser = nn.layer_count_traverser
nn_traverser = nn.nn_traverser

nn_compose = nn.compose

## Torch

torch_traverser = lambda x: x  # noqa: E731
nn_traverser.register("torch.nn.modules.module.Module", torch_traverser)
