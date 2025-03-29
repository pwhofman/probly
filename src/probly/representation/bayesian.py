import copy

import torch
import torch.nn as nn

from .layers import BayesConv2d, BayesLinear


class Bayesian(nn.Module):
    """
    This class implements a dropout model to be used for uncertainty quantification.
    Args:
        base: torch.nn.Module, The base model.

    Attributes:
        model: torch.nn.Module, The transformed model with Bayesian layers.
    """
    def __init__(self,
                 base: nn.Module,
                 posterior_std: float=0.05,
                 prior_mean: float=0.0,
                 prior_std: float=1.0):
        super().__init__()
        self._convert(base, posterior_std, prior_mean, prior_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def represent_uncertainty(self, x: torch.Tensor, n_samples: int=25) -> torch.Tensor:
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)

    def _convert(self,
                 base: nn.Module,
                 posterior_std: float,
                 prior_mean: float,
                 prior_std: float) -> None:
        """
        Converts the base model to a Bayesian model, stored in model, by replacing all layers by
        Bayesian layers.
        Args:
            base: torch.nn.Module, The base model to be used for dropout.
            posterior_std: float, The posterior standard deviation.
            prior_mean: float, The prior mean.
            prior_std: float, The prior standard deviation.
        """
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
    def kl_divergence(self) -> torch.Tensor:
        """
        Collects the KL divergence of the model by summing the KL divergence of each layer.
        """
        kl = 0
        for module in self.model.modules():
            if isinstance(module, BayesLinear | BayesConv2d):
                kl += module.kl_divergence
        return kl
