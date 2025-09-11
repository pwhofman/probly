"""Dropout ensemble implementation for uncertainty quantification."""

from __future__ import annotations

import torch
from torch import nn

from probly.representation.predictor_torch import TorchSamplingRepresentationPredictor
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, singledispatch_traverser, traverse

from . import Dropout, register


def _prepend_torch_dropout(obj: nn.Module, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(p=p), obj)


@singledispatch_traverser[object]
def _eval_dropout_traverser(obj: TorchDropout) -> TorchDropout:
    """Ensure that Dropout layers are active during evaluation."""
    return obj.train()


class TorchDropout[In, KwIn](Dropout[In, KwIn, torch.Tensor], TorchSamplingRepresentationPredictor[In, KwIn]):
    """Implementation of a Dropout ensemble class to be used for uncertainty quantification."""

    def eval(self) -> TorchDropout:
        """Sets the model to evaluation mode but keeps the dropout layers active."""
        super().eval()

        traverse(self.model, nn_compose(_eval_dropout_traverser), init={CLONE: False})

        return self


register(nn.Linear, _prepend_torch_dropout)
