"""Torch subensemble implementation."""

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


def generate_torch_subensemble(
    obj: nn.Module,
    num_heads: int,
    reset_params: bool = False,
) -> nn.ModuleList:
    """Build a torch subensemble by copying the last layer num_heads times, resetting the parameters of each head."""
    feature_extractor = nn.Sequential(*list(obj.children())[:-1])
    last_layer = list(obj.children())[-1]

    if reset_params:
        heads = nn.ModuleList([_reset_copy(last_layer) for _ in range(num_heads)])
    else:
        heads = nn.ModuleList([_copy(last_layer) for _ in range(num_heads)])

    return nn.ModuleList([feature_extractor, heads])


register(nn.Module, generate_torch_subensemble)
