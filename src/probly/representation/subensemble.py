"""SubEnsemble class implementation."""

from __future__ import annotations

from typing import Unpack

import torch
from torch import nn
from torch.nn import functional as F

from probly.representation.predictor import Predictor, RepresentationPredictor
from probly.traverse_nn import TORCH_CLONE, nn_compose
from pytraverse import (
    singledispatch_traverser,
    traverse,
)

reset_traverser = singledispatch_traverser[object](name="reset_traverser")


@reset_traverser.register
def _(obj: nn.Module) -> nn.Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()
    return obj


def _reset_copy(module: nn.Module) -> nn.Module:
    return traverse(module, nn_compose(reset_traverser), init={TORCH_CLONE: True})


def convert(base: nn.Module | None, n_heads: int, head: nn.Module) -> nn.ModuleList:
    """Convert a model into an ensemble with trainable heads.

    Args:
        base: torch.nn.Module, The base model to be used.
        n_heads: int, The number of heads in the ensemble.
        head: torch.nn.Module, The head to be used. Can be a complete network or a single layer.
    """
    if base is not None:
        for param in base.parameters():
            param.requires_grad = False

    return nn.ModuleList(
        [(nn.Sequential(base, _reset_copy(head)) if base is not None else _reset_copy(head)) for _ in range(n_heads)],
    )


class SubEnsemble[In, KwIn](
    nn.Module,
    Predictor[In, KwIn, torch.Tensor],
    RepresentationPredictor[
        In,
        KwIn,
        torch.Tensor,
        torch.Tensor,
    ],
):
    """Ensemble class of members with shared, frozen backbone and trainable heads.

    This class implements an ensemble of models which share a backbone and use
    different heads that can be made up of multiple layers.
    The backbone is frozen and only the head can be trained.

    Attributes:
        models: torch.nn.ModuleList, The list of models in the ensemble consisting of the frozen
        base model and the trainable heads.

    """

    def __init__(self, base: nn.Module | None, n_heads: int, head: nn.Module) -> None:
        """Convert a model into an ensemble with trainable heads.

        Args:
            base: torch.nn.Module, The base model to be used.
            n_heads: int, The number of heads in the ensemble.
            head: torch.nn.Module, The head to be used. Can be a complete network or a single layer.
        """
        super().__init__()
        self.models = convert(base, n_heads, head)

    def forward(
        self,
        *args: In,
        logits: bool = False,
        **kwargs: Unpack[KwIn],
    ) -> torch.Tensor:
        """Forward pass of the ensemble."""
        return self.predict_pointwise(*args, logits=logits, **kwargs)

    def predict_pointwise(
        self,
        *args: In,
        logits: bool = False,
        **kwargs: Unpack[KwIn],
    ) -> torch.Tensor:
        """Forward pass that gives a point-wise prediction."""
        if logits:
            return torch.stack(
                [model(*args, **kwargs) for model in self.models],
                dim=1,
            ).mean(dim=1)
        return torch.stack(
            [F.softmax(model(*args, **kwargs), dim=1) for model in self.models],
            dim=1,
        ).mean(dim=1)

    def predict_representation(
        self,
        *args: In,
        logits: bool = False,
        **kwargs: Unpack[KwIn],
    ) -> torch.Tensor:
        """Forward pass that gives an uncertainty representation."""
        if logits:
            return torch.stack([model(*args, **kwargs) for model in self.models], dim=1)
        return torch.stack(
            [F.softmax(model(*args, **kwargs), dim=1) for model in self.models],
            dim=1,
        )
