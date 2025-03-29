import copy

import torch
import torch.nn as nn

from .layers import BayesLinear


class Bayesian(nn.Module):
    def __init__(self, base):
        super().__init__()
        self._convert(base)

    def forward(self, x):
        return self.model(x)

    def represent_uncertainty(self, x, n_samples=25):
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)


    def _convert(self, base):
        self.model = copy.deepcopy(base)
        self.n_parameters = 0
        for name, child in self.model.named_children():
            if isinstance(child, nn.Linear):
                setattr(self.model, name, BayesLinear(child.in_features, child.out_features))

    @property
    def kl_divergence(self):
        kl = 0
        for module in self.model.modules():
            if isinstance(module, BayesLinear):
                kl += module.kl_divergence
        return kl
