"""Torch subensemble implementation."""

from __future__ import annotations

from torch import nn

from probly.transformation.ensemble import ensemble

from .common import register


def generate_torch_subensemble(
    obj: nn.Module,
    num_heads: int,
    *,
    head: nn.Module | None = None,
    reset_params: bool = False,
    head_layer: int = 1,
) -> nn.ModuleList:
    """Build a torch subensemble.

    By either:
    - copying head_layer (default: 1) last layers of obj num_heads times, if no head model is provided.
    - using an obj as shared backbone and copying the head model num_heads times.
    Resets the parameters of each head.
    """
    # no head
    if head is None:
        head = nn.Sequential(*list(obj.children())[-head_layer:])
        obj = nn.Sequential(*list(obj.children())[:-head_layer])

    # obj and head
    heads = ensemble(head, num_members=num_heads, reset_params=reset_params)

    return nn.ModuleList([obj, heads])


register(nn.Module, generate_torch_subensemble)
