import torch
import torch.nn as nn

class Ensemble(nn.Module):
    """
    This class implements an ensemble of models to be used for uncertainty quantification.

    Args:
    base (torch.nn.Module): The base model to be used.
    num_members (int): The number of members in the ensemble.
    """
    def __init__(self, base, num_members):
        super(Ensemble, self).__init__()
        self.base = base
        self.models = nn.ModuleList([self.base for _ in range(num_members)])

    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=1)
