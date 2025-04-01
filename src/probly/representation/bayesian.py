"""Bayesian Neural Network (BNN) implementation."""

import copy

import torch
from torch import nn

from probly.representation.layers import BayesConv2d, BayesLinear


class Bayesian(nn.Module):
    """Implementation of a Bayesian neural network to be used for uncertainty quantification.

    Implementation is based on https://arxiv.org/abs/1505.05424.

    Attributes:
        model: torch.nn.Module, The transformed model with Bayesian layers.
    """

    def __init__(
        self,
        base: nn.Module,
        use_base_weights: bool = False,
        posterior_std: float = 0.05,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> None:
        """Initialize an instance of the Bayesian class.

        Convert the base model into a Bayesian model by replacing suitable layers by Bayesian layers.

        Args:
            base: torch.nn.Module, The base model.
            use_base_weights: bool, If True, the weights of the base model are used as the prior mean.
            posterior_std: float, The initial posterior standard deviation.
            prior_mean: float, The prior mean.
            prior_std: float, The prior standard deviation.
        """
        super().__init__()
        self._convert(base, use_base_weights, posterior_std, prior_mean, prior_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Bayesian model.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output
        """
        return self.model(x)

    def represent_uncertainty(self, x: torch.Tensor, n_samples: int = 25) -> torch.Tensor:
        """Forward pass that gives an uncertainty representation.

        Args:
            x: torch.Tensor, input data
            n_samples: int, number of samples to draw from posterior
        Returns:
            torch.Tensor, uncertainty representation, samples from the posterior
        """
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)

    def _convert(
        self,
        base: nn.Module,
        use_base_weights: bool,
        posterior_std: float,
        prior_mean: float,
        prior_std: float,
    ) -> None:
        """Converts the base model to a Bayesian model by replacing all layers by Bayesian layers.

        Args:
            base: torch.nn.Module, The base model to be used for dropout.
            use_base_weights: bool, If True, the weights of the base model are used as the prior mean.
            posterior_std: float, The posterior standard deviation.
            prior_mean: float, The prior mean.
            prior_std: float, The prior standard deviation.
        """
        self.model = copy.deepcopy(base)

        def apply_bayesian(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(
                        module,
                        name,
                        BayesLinear(
                            child.in_features,
                            child.out_features,
                            child.bias is not None,
                            posterior_std,
                            prior_mean,
                            prior_std,
                            child if use_base_weights else None,
                        ),
                    )
                elif isinstance(child, nn.Conv2d):
                    setattr(
                        module,
                        name,
                        BayesConv2d(
                            child.in_channels,
                            child.out_channels,
                            child.kernel_size,
                            child.stride,
                            child.padding,
                            child.dilation,
                            child.groups,
                            child.bias is not None,
                            posterior_std,
                            prior_mean,
                            prior_std,
                            child if use_base_weights else None,
                        ),
                    )
                else:
                    apply_bayesian(child)  # apply recursively to all layers

        apply_bayesian(self.model)

    @property
    def kl_divergence(self) -> torch.Tensor:
        """Collects the KL divergence of the model by summing the KL divergence of each layer."""
        kl = 0
        for module in self.model.modules():
            if isinstance(module, BayesLinear | BayesConv2d):
                kl += module.kl_divergence
        return kl
