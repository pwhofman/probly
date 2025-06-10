"""Protocols and ABCs for Torch representation wrappers."""

from __future__ import annotations

from abc import ABCMeta

import torch
from torch import nn

from probly.representation.predictor import RepresentationPredictorWrapper


class TorchRepresentationPredictorWrapper[KwIn](
    RepresentationPredictorWrapper[
        torch.Tensor,
        KwIn,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    nn.Module,
    # metaclass=ABCMeta,
):
    """Abstract base class for Torch representation predictors."""
