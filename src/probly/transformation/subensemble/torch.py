"""Torch subensemble implementation."""

from __future__ import annotations

from torch import nn

from probly.transformation.ensemble import ensemble
from pytraverse import singledispatch_traverser, traverse

from .common import register

subensemble_traverser = singledispatch_traverser[nn.Module](name="subensemble_traverser")


@subensemble_traverser.register
def _(obj: nn.Module) -> nn.Module:
    return list(obj.children())


def generate_torch_subensemble(
    obj: nn.Module,
    num_heads: int,
    *,
    head: nn.Module | None = None,
    reset_params: bool = False,
    head_layer: int,
) -> nn.ModuleList:
    """Build a torch subensemble.

    By either:
    - copying head_layer (default: 1) last layers of obj num_heads times, if no head model is provided.
    - using an obj as shared backbone and copying the head model num_heads times.
    Resets the parameters of each head.
    """
    layers = [m for m in traverse(obj, subensemble_traverser) if isinstance(m, nn.Module)]

    if head_layer > len(layers):
        msg = f"head_layer {head_layer} must be less than to {len(layers)}"
        raise ValueError(msg)

    # no head
    if head is None:
        backbone = nn.Sequential(*layers[:-head_layer])
        head = nn.Sequential(*layers[-head_layer:])
    else:
        backbone = obj

    # save head to device
    device = next(obj.parameters()).device
    head.to(device)

    # call ensemble to create heads from head
    heads = ensemble(head, num_members=num_heads, reset_params=reset_params)
    heads.to(device)

    # freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    backbone_layers = [m for m in traverse(backbone, subensemble_traverser) if isinstance(m, nn.Module)]

    subensemble = nn.ModuleList(
        [
            nn.Sequential(
                nn.Sequential(*backbone_layers),
                nn.Sequential(*[m for m in traverse(h, subensemble_traverser) if isinstance(m, nn.Module)]),
            )
            for h in heads
        ],
    )

    return subensemble


register(nn.Module, generate_torch_subensemble)
