from ..utils import torch_reset_all_parameters

import copy

import torch
import torch.nn as nn

class Ensemble(nn.Module):
    """
    This class implements an ensemble of representation to be used for uncertainty quantification.

    Args:
    base (torch.nn.Module): The base model to be used.
    num_members (int): The number of members in the ensemble.
    """
    def __init__(self, base, num_members):
        super(Ensemble, self).__init__()
        self._convert(base, num_members)

    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=1)

    def _convert(self, base, num_members):
        self.models = nn.ModuleList()
        for _ in range(num_members):
            model = copy.deepcopy(base)
            torch_reset_all_parameters(model)
            self.models.append(model)
