"""Torch dropout implementation."""

from __future__ import annotations

from torch import nn

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse

from .common import register

reset_traverser = lazydispatch_traverser[object](name="reset_traverser")


@reset_traverser.register
def _(obj: nn.Module) -> nn.Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()  # type: ignore[operator]
    return obj


def _reset_copy(module: nn.Module) -> nn.Module:
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def generate_torch_ensemble(obj: nn.Module, n_members: int) -> nn.ModuleList:
    """Build a torch ensemble by copying the base model n_members times."""
    return nn.ModuleList([_reset_copy(obj) for _ in range(n_members)])


register(nn.Module, generate_torch_ensemble)
