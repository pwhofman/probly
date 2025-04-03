"""SubEnsemble class implementation."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from probly.utils import torch_reset_all_parameters


class SubEnsemble(nn.Module):
    """Ensemble class of members with shared, frozen backbone and trainable heads.

    This class implements an ensemble of models which share a backbone and use
    different heads that can be made up of multiple layers.
    The backbone is frozen and only the head can be trained.

    Attributes:
        models: torch.nn.ModuleList, The list of models in the ensemble consisting of the frozen
        base model and the trainable heads.

    """

    def __init__(self, base: nn.Module, n_members: int, head: nn.Module) -> None:
        """Initialization of an instance of the SubEnsemble class.

        Ensemble members are created by taking the frozen base model and appending a head model.

        Args:
            base: torch.nn.Module, The base model to be used.
            n_members: int, The number of members in the ensemble.
            head: torch.nn.Module, The head to be used. Can be a complete network or a single layer.

        """
        super().__init__()
        self._convert(base, n_members, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sub-ensemble.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output

        """
        return torch.stack([model(x) for model in self.models], dim=1).mean(dim=1)

    def predict_pointwise(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        """Forward pass that gives a point-wise prediction.

        Args:
            x: torch.Tensor, input data
            logits: bool, whether to return logits or probabilities
        Returns:
            torch.Tensor, point-wise prediction
        """
        if logits:
            return torch.stack([model(x) for model in self.models], dim=1).mean(dim=1)
        return torch.stack([F.softmax(model(x), dim=1) for model in self.models], dim=1).mean(dim=1)

    def predict_representation(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        """Forward pass that gives an uncertainty representation.

        Args:
            x: torch.Tensor, input data
            logits: bool, whether to return logits or probabilities
        Returns:
            torch.Tensor, uncertainty representation
        """
        if logits:
            return torch.stack([model(x) for model in self.models], dim=1)
        return torch.stack([F.softmax(model(x), dim=1) for model in self.models], dim=1)

    def _convert(self, base: nn.Module, n_heads: int, head: nn.Module) -> None:
        """Convert a model into an ensemble with trainable heads.

        Args:
            base: torch.nn.Module, The base model to be used.
            n_heads: int, The number of heads in the ensemble.
            head: torch.nn.Module, The head to be used. Can be a complete network or a single layer.

        """
        for param in base.parameters():
            param.requires_grad = False
        self.models = nn.ModuleList()
        for _ in range(n_heads):
            h = copy.deepcopy(head)
            torch_reset_all_parameters(h)
            self.models.append(nn.Sequential(base, h))
