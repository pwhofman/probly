"""Dropout ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from torch import nn

from probly.traverse import (
    singledispatch_traverser,
)
from probly.traverse_nn import is_first_layer

from .drop import Drop, P

dropout_traverser = singledispatch_traverser(name="dropout_traverser")


def _prepend_dropout(obj: nn.Module, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(p=p), obj)


def register(cls: type) -> None:
    """Register a class to be prepended by Dropout layers."""
    dropout_traverser.register(
        cls,
        _prepend_dropout,
        skip_if=is_first_layer,
        vars={"p": P},
    )


@singledispatch_traverser
def _eval_dropout_traverser(obj: nn.Dropout) -> nn.Dropout:
    """Ensure that Dropout layers are active during evaluation."""
    return obj.train()


class Dropout(Drop):
    """Implementation of a Dropout ensemble class to be used for uncertainty quantification.

    Attributes:
        p: float, The probability of dropout.
        model: torch.nn.Module, The model with Dropout layers.

    """

    _convert_traverser = dropout_traverser
    _eval_traverser = _eval_dropout_traverser


register(nn.Linear)
