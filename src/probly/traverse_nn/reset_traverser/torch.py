"""Torch implementation of reset traverser."""

from __future__ import annotations

from torch import nn

from ._common import reset_traverser


@reset_traverser.register(cls=nn.Module)
def _(obj: nn.Module) -> nn.Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()  # ty: ignore[call-non-callable]
    return obj
