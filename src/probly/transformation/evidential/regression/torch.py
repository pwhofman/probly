"""Torch evidential regression implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from probly.layers.torch import NormalInverseGammaLinear
from probly.transformation.evidential.regression.common import REPLACED_LAST_LINEAR, register

if TYPE_CHECKING:
    from pytraverse import State, TraverserResult


def replace_last_torch_nig(obj: nn.Linear, state: State) -> TraverserResult:
    """Register a class to be replaced by the NormalInverseGammaLinear layer."""
    state[REPLACED_LAST_LINEAR] = True
    return NormalInverseGammaLinear(
        obj.in_features,
        obj.out_features,
        device=obj.weight.device,
        bias=obj.bias is not None,
    ), state


register(nn.Linear, replace_last_torch_nig)
