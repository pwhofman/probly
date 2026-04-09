"""Torch dare implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from probly.layers.torch import DAREWrapper
from probly.traverse_nn import nn_compose
from pytraverse import GlobalVariable, singledispatch_traverser, traverse

from ._common import dare_generator

if TYPE_CHECKING:
    from pytraverse.core import State

dare_traverser = singledispatch_traverser[nn.Module](name="dare_traverser")
NUM_MEMBERS = GlobalVariable[int]("NUM_MEMBERS")
DELTA = GlobalVariable[float]("DELTA")


@dare_traverser.register
def _(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    valid_layer_types = (nn.Conv2d, nn.Linear)
    in_attr = "in_features" if hasattr(obj, "in_features") else "in_channels" if hasattr(obj, "in_channels") else None
    out_attr = (
        "out_features" if hasattr(obj, "out_features") else "out_channels" if hasattr(obj, "out_channels") else None
    )

    if in_attr and out_attr and isinstance(obj, valid_layer_types):
        return DAREWrapper(
            obj,
            delta=DELTA(state),
            num_members=NUM_MEMBERS(state),
            in_attr=getattr(obj, in_attr),
            out_attr=getattr(obj, out_attr),
        ), state
    return obj, state


@dare_generator.register(nn.Module)
def torch_dare_wrapper(obj: nn.Module, num_members: int, delta: float) -> nn.Module:
    """Apply dare wrapper."""
    return traverse(obj, nn_compose(dare_traverser), init={NUM_MEMBERS: num_members, DELTA: delta})
