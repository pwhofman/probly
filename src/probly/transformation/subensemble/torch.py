"""Torch subensemble implementation."""

from __future__ import annotations

from torch import nn

from probly.transformation.ensemble import ensemble
from probly.traverse_nn import nn_traverser
from pytraverse import traverse

from .common import register


def generate_torch_subensemble(
    obj: nn.Module,
    num_heads: int,
    *,
    head: nn.Module | None = None,
    reset_params: bool = False,
    head_layer: int | None = 1,
) -> nn.ModuleList:
    """Build a torch subensemble.

    By either:
    - copying head_layer (default: 1) last layers of obj num_heads times, if no head model is provided.
    - using an obj as shared backbone and copying the head model num_heads times.
    Resets the parameters of each head.
    """
    if head is None:
        if head_layer is None:
            msg = "head_layer must be provided when head is not provided."
            raise ValueError(msg)
        layers = [m for m in traverse(obj, nn_traverser).children() if isinstance(m, nn.Module)]
        if not isinstance(obj, nn.Sequential):
            msg = f"head_layer is only supported for nn.Sequential models, but got {type(obj)} instead."
            raise ValueError(msg)
        if head_layer > len(layers):
            msg = f"head_layer {head_layer} must be less than to {len(layers)}"
            raise ValueError(msg)

        backbone = nn.Sequential(*layers[:-head_layer])
        head = nn.Sequential(*layers[-head_layer:])
    else:
        backbone = obj

    # save head to device
    device = next(obj.parameters()).device
    head.to(device)

    # call ensemble to create heads from head
    heads = ensemble(head, num_members=num_heads, reset_params=reset_params)

    # freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    subensemble = nn.ModuleList(
        [
            nn.Sequential(
                backbone,
                head,
            )
            for head in heads  # ty:ignore[not-iterable]
        ],
    )

    return subensemble


register(nn.Module, generate_torch_subensemble)
