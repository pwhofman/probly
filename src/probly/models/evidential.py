import copy
import torch
import torch.nn as nn

class Evidential(nn.Module):
    """
    This class implements an evidential deep learning model to be used for uncertainty quantification.

    Args:
    base (torch.nn.Module): The base model to be used.
    num_members (int): The number of members in the ensemble.
    """
    def __init__(self, base, activation=nn.Softplus()):
        super(Evidential, self).__init__()
        self.base = base
        self.model = None
        self._convert(base, activation)

    def forward(self, x):
        return self.model(x)

    def _convert(self, base, activation):
        self.model = nn.Sequential(*list(base.children()), activation)

    def sample(self, x, num_samples):
        dirichlet = torch.distributions.Dirichlet(self.model(x))
        return torch.stack([dirichlet.sample() for _ in range(num_samples)]).swapaxes(0, 1)




