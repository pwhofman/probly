"""Torch subensemble implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.method.ensemble import ensemble
from probly.traverse_nn import nn_traverser
from pytraverse import traverse

from ._common import subensemble_generator


class _FrozenBackbone(nn.Module):
    """Backbone that stays in eval mode regardless of ``model.train()`` calls."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            p.requires_grad = False
        self.module.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(x)

    def train(self, mode: bool = True) -> _FrozenBackbone:  # noqa: ARG002
        super().train(False)
        return self


@subensemble_generator.register(nn.Module)
def generate_torch_subensemble(
    obj: nn.Module,
    num_heads: int,
    *,
    head: nn.Module | None = None,
    reset_params: bool = True,
    head_layer: int | None = 1,
) -> nn.ModuleList:
    """Build a torch subensemble. See :func:`probly.method.subensemble.subensemble`."""
    if head is None:
        if head_layer is None:
            msg = "head_layer must be provided when head is not provided."
            raise ValueError(msg)
        if not isinstance(obj, nn.Sequential):
            msg = (
                f"head_layer slicing is only supported for nn.Sequential models, "
                f"but got {type(obj).__name__}. For non-Sequential models, pass "
                "an explicit head module via the `head` argument."
            )
            raise ValueError(msg)
        layers = [m for m in traverse(obj, nn_traverser).children() if isinstance(m, nn.Module)]
        if head_layer > len(layers):
            msg = f"head_layer {head_layer} must be less than to {len(layers)}"
            raise ValueError(msg)

        backbone: nn.Module = nn.Sequential(*layers[:-head_layer])
        head = nn.Sequential(*layers[-head_layer:])
    else:
        backbone = obj

    device = next(obj.parameters()).device
    head.to(device)

    heads = ensemble(head, num_members=num_heads, reset_params=reset_params)
    frozen_backbone = _FrozenBackbone(backbone)

    return nn.ModuleList([nn.Sequential(frozen_backbone, h) for h in heads])
