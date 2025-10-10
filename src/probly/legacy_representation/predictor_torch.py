"""Protocols and ABCs for Torch representation wrappers."""

from __future__ import annotations

from abc import ABCMeta
from typing import Unpack

import torch
from torch import nn
from torch.nn import functional as F

from probly.predictor import PredictorConverter, SamplingRepresentationPredictor


class TorchSamplingRepresentationPredictor[In, KwIn](
    nn.Module,
    SamplingRepresentationPredictor[
        In,
        KwIn,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    PredictorConverter[nn.Module],
    metaclass=ABCMeta,
):
    """Abstract class for PyTorch-based sampling representation predictors."""

    def forward(self, *args: In, logits: bool = False, **kwargs: Unpack[KwIn]) -> torch.Tensor:
        """Forward pass of the model."""
        res: torch.Tensor = self.model(*args, **kwargs)

        if not logits:
            return F.softmax(res, dim=1)

        return res

    def _create_representation(self, y: list[torch.Tensor]) -> torch.Tensor:
        """Create a representation from a collection of outputs."""
        return torch.stack(y, dim=1)

    def _create_pointwise(self, y: list[torch.Tensor]) -> torch.Tensor:
        """Create a pointwise output from a collection of outputs."""
        return torch.stack(y, dim=1).mean(dim=1)
