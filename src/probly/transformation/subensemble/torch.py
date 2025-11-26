"""Torch subensemble implementation."""

from __future__ import annotations

from torch import nn

from probly.transformation.ensemble import ensemble

from .common import register


def generate_torch_subensemble(
    obj: nn.Module,
    num_heads: int,
    head: nn.Module | None = None,
    reset_params: bool = False,
    head_layer: int = 1,
) -> nn.ModuleList:
    """Build a torch subensemble by copying the last layer or head model num_heads times.

    Resets the parameters of each head.
    """
    if head is None:
        head = nn.Sequential(*list(obj.children())[-head_layer:])
        obj = nn.Sequential(*list(obj.children())[:-head_layer])

    if reset_params:
        heads = ensemble(head, num_members=num_heads, reset_params=reset_params)
    else:
        heads = ensemble(head, num_members=num_heads, reset_params=reset_params)

    return nn.ModuleList([obj, heads])


register(nn.Module, generate_torch_subensemble)
