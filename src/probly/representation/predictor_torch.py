"""Protocols and ABCs for Torch representation wrappers."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Unpack

import torch
from torch import nn
from torch.nn import functional as F

from probly.representation.predictor import SamplingRepresentationPredictor


class TorchSamplingRepresentationPredictor[In, KwIn](
    nn.Module,
    SamplingRepresentationPredictor[
        In,
        KwIn,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    metaclass=ABCMeta,
):
    """Abstract class for PyTorch-based sampling representation predictors."""

    def __init__(self, base: nn.Module) -> None:
        """Initialize the predictor with a base model."""
        super().__init__()
        self.model = self._convert(base)

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

    @abstractmethod
    def _convert(self, base: nn.Module) -> nn.Module:
        """Convert the base model to a representation model."""
        return base
