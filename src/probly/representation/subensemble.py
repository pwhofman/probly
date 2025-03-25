import copy

import torch
import torch.nn as nn

from ..utils import torch_reset_all_parameters


class SubEnsemble(nn.Module):
    """
    This class implements an ensemble of representation which share a backbone and use
    different classification heads that can be made up of multiple layers.
    The backbone is frozen and only the head can be trained.
    Args:
        base: torch.nn.Module, The base model to be used.
        n_heads: int, The number of heads in the ensemble.
        head: torch.nn.Module, The classification head to be used. Can be a complete network or a single layer.
    """

    def __init__(self, base: nn.Module, n_heads: int, head: nn.Module) -> None:
        super().__init__()
        self._convert(base, n_heads, head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sub-ensemble.
        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output
        """
        return torch.stack([model(x) for model in self.models], dim=1)

    def _convert(self, base: nn.Module, n_heads: int, head: nn.Module) -> None:
        """
        Convert a model into an ensemble with trainable heads.
        Args:
            base: torch.nn.Module, The base model to be used.
            n_heads: int, The number of heads in the ensemble.
            head: torch.nn.Module, The classification heads to be used. Can be a complete network or a single layer.
        """
        for param in base.parameters():
            param.requires_grad = False
        self.models = nn.ModuleList()
        for _ in range(n_heads):
            h = copy.deepcopy(head)
            torch_reset_all_parameters(h)
            self.models.append(nn.Sequential(base, h))
