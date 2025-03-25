import copy

import torch
import torch.nn as nn

from ..utils import torch_reset_all_parameters


class Ensemble(nn.Module):
    """
    This class implements an ensemble of representation to be used for uncertainty quantification.
    Args:
        base: torch.nn.Module, The base model to be used.
        n_members: int, The number of members in the ensemble.
    """

    def __init__(self, base: nn.Module, n_members: int) -> None:
        super().__init__()
        self._convert(base, n_members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ensemble.
        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, ensemble output
        """
        return torch.stack([model(x) for model in self.models], dim=1)

    def _convert(self, base: nn.Module, n_members: int) -> None:
        """
        Convert the base model to an ensemble with n_members members.
        Args:
            base: torch.nn.Module, The base model to be converted.
            n_members: int, The number of members in the ensemble.
        """
        self.models = nn.ModuleList()
        for _ in range(n_members):
            model = copy.deepcopy(base)
            torch_reset_all_parameters(model)
            self.models.append(model)
