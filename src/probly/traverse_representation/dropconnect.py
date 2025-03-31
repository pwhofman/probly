import torch.nn as nn

from probly.representation.layers import DropConnectLinear
from probly.traverse import (
    singledispatch_traverser,
)
from probly.traverse_nn import is_first_layer

from .drop import Drop, P

dropconnect_traverser = singledispatch_traverser(name="dropout_traverser")


@dropconnect_traverser.register(skip_if=is_first_layer, vars=dict(p=P))
def _(obj: nn.Linear, p: float):
    return DropConnectLinear(obj, p)


@singledispatch_traverser
def eval_traverser(obj: DropConnectLinear):
    """Ensure that DropConnect layers are active during evaluation."""
    return obj.train()


class DropConnect(Drop):
    """
    This class implements DropConnect to be used for uncertainty quantification
    based on http://proceedings.mlr.press/v28/wan13.pdf.
    Args:
        base: torch.nn.Module, The base model to be used for DropConnect.
        p: float, The probability of dropping out individual weights.

    Attributes:
        p: float, The probability of dropping out individual weights.
        model: torch.nn.Module, The transformed model with DropConnect layers.
    """

    _convert_traverser = dropconnect_traverser
    _eval_traverser = eval_traverser
