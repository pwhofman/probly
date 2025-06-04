import torch.nn as nn

from probly.representation.layers import DropConnectLinear
from probly.traverse import (
    singledispatch_traverser,
)
from probly.traverse_nn import is_first_layer

from .drop import Drop, P

dropconnect_traverser = singledispatch_traverser(name="dropconnect_traverser")


@dropconnect_traverser.register(skip_if=is_first_layer, vars={"p": P})
def _(obj: nn.Linear, p: float) -> DropConnectLinear:
    return DropConnectLinear(obj, p)


@singledispatch_traverser
def eval_traverser(obj: DropConnectLinear) -> DropConnectLinear:
    """Ensure that DropConnect layers are active during evaluation."""
    return obj.train()


class DropConnect(Drop):
    """Implementation of a DropConnect model to be used for uncertainty quantification.

    Implementation is based on https://proceedings.mlr.press/v28/wan13.pdf.

    Attributes:
        p: float, the probability of dropping out individual weights.
        model: torch.nn.Module, the model with DropConnect layers.

    """

    _convert_traverser = dropconnect_traverser
    _eval_traverser = eval_traverser
