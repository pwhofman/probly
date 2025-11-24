"""Torch dropout implementation."""

from __future__ import annotations

from torch import nn

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from .common import register

reset_traverser = singledispatch_traverser[nn.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nn.Module) -> nn.Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()  # type: ignore[operator]
    return obj


def _reset_copy(module: nn.Module) -> nn.Module:
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def _copy(module: nn.Module) -> nn.Module:
    return traverse(module, nn_traverser, init={CLONE: True})


def generate_torch_ensemble(
    obj: nn.Module,
    num_members: int,
    reset_params: bool = True,
) -> nn.ModuleList:
    """Build a torch ensemble by copying the base model num_members times, resetting the parameters of each member."""
    if reset_params:
        return nn.ModuleList([_reset_copy(obj) for _ in range(num_members)])
    return nn.ModuleList([_copy(obj) for _ in range(num_members)])


register(nn.Module, generate_torch_ensemble)
