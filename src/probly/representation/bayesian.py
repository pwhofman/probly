import copy

import torch
import torch.nn as nn

from .layers import BayesConv2d, BayesLinear


class Bayesian(nn.Module):
    def __init__(self, base, posterior_std=0.05, prior_mean=0.0, prior_std=1.0):
        super().__init__()
        self._convert(base, posterior_std, prior_mean, prior_std)

    def forward(self, x):
        return self.model(x)

    def represent_uncertainty(self, x, n_samples=25):
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)

    def _convert(self, base, posterior_std, prior_mean, prior_std):
        self.model = copy.deepcopy(base)
        self.n_parameters = 0
        for name, child in self.model.named_children():
            if isinstance(child, nn.Linear):
                setattr(self.model, name, BayesLinear(child.in_features,
                                                      child.out_features,
                                                      child.bias is not None,
                                                      posterior_std,
                                                      prior_mean,
                                                      prior_std))
            elif isinstance(child, nn.Conv2d):
                setattr(self.model, name, BayesConv2d(child.in_channels,
                                                      child.out_channels,
                                                      child.kernel_size,
                                                      child.stride,
                                                      child.padding,
                                                      child.dilation,
                                                      child.groups,
                                                      child.bias is not None,
                                                      posterior_std,
                                                      prior_mean,
                                                      prior_std))


    @property
    def kl_divergence(self):
        kl = 0
        for module in self.model.modules():
            if isinstance(module, BayesLinear | BayesConv2d):
                kl += module.kl_divergence
        return kl
