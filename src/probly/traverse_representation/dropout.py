import torch.nn as nn

from probly.traverse import (
    singledispatch_traverser,
)
from probly.traverse_nn import is_first_layer

from .drop import Drop, P

dropout_traverser = singledispatch_traverser(name="dropout_traverser")


def prepend_dropout(obj: nn.Module, p: float):
    return nn.Sequential(nn.Dropout(p=p), obj)


def register(cls: type):
    dropout_traverser.register(
        cls, prepend_dropout, skip_if=is_first_layer, vars=dict(p=P)
    )


@singledispatch_traverser
def eval_traverser(obj: nn.Dropout):
    """Ensure that Dropout layers are active during evaluation."""
    return obj.train()


class Dropout(Drop):
    """
    This class implements a dropout layer to be used for uncertainty quantification.
    Args:
        base: torch.nn.Module, The base model to be used for dropout.
        p: float, The probability of dropping out a neuron.

    Attributes:
        p: float, The probability of dropout.
        model: torch.nn.Module, The transformed model with Dropout layers.
    """

    _convert_traverser = dropout_traverser
    _eval_traverser = eval_traverser


register(nn.Linear)
