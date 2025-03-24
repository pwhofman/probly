import copy

import torch
import torch.nn as nn

class Evidential(nn.Module):
    """
    This class implements an evidential deep learning model to be used for uncertainty quantification.

    Args:
    base (torch.nn.Module): The base model to be used.
    activation (torch.nn.Module): The activation function that will be used.
    """
    def __init__(self, base, activation=nn.Softplus()):
        super().__init__()
        self._convert(base, activation)

    def forward(self, x):
        return self.model(x)

    def _convert(self, base, activation):
        self.model = nn.Sequential(copy.deepcopy(base), activation)

    def sample(self, x, num_samples):
        dirichlet = torch.distributions.Dirichlet(self.model(x) + 1.0)
        return torch.stack([dirichlet.sample() for _ in range(num_samples)]).swapaxes(0, 1)
