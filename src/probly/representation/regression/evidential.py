"""Evidential model class implementation."""

from __future__ import annotations

import copy

import torch
from torch import nn

from probly.representation.layers import NormalInverseGammaLinear


class Evidential(nn.Module):
    """This class implements an evidential deep learning model for regression.

    Attributes:
        model: torch.nn.Module, The evidential model with a normal inverse gamma layer suitable
        for evidential regression.

    """

    def __init__(self, base: nn.Module) -> None:
        """Initialize the Evidential model.

        Convert the base model into an evidential deep learning regression model.

        Args:
            base: torch.nn.Module, The base model to be used.
        """
        super().__init__()
        self._convert(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output

        """
        return self.model(x)

    def predict_pointwise(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model for point-wise prediction.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output
        """
        return self.model(x)

    def predict_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model for uncertainty representation.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output

        """
        return self.model(x)

    def _convert(self, base: nn.Module) -> None:
        """Convert a model into an evidential deep learning regression model.

        Replace the last layer by a layer parameterizing a normal inverse gamma distribution.

        Args:
            base: torch.nn.Module, The base model to be used.

        """
        self.model = copy.deepcopy(base)
        for name, child in reversed(list(self.model.named_children())):
            if isinstance(child, nn.Linear):
                setattr(
                    self.model,
                    name,
                    NormalInverseGammaLinear(child.in_features, child.out_features),
                )
                break

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample from the predicted distribution for a given input x.

        Returns a tensor of shape (n_instances, n_samples, 2) representing the parameters of the sampled normal
        distributions. The mean of the normal distribution is the gamma parameter and the variance is sampled from
        an inverse gamma distribution and divided by the nu parameter. The first dimension is the mean and
        the second dimension is the variance.

        Args:
            x: torch.Tensor, input data
            n_samples: int, number of samples
        Returns:
            torch.Tensor, samples

        """
        x = self.model(x)
        inverse_gamma = torch.distributions.InverseGamma(x["alpha"], x["beta"])
        sigma2 = torch.stack([inverse_gamma.sample() for _ in range(n_samples)]).swapaxes(0, 1)
        normal_mu = x["gamma"].unsqueeze(-1).expand(-1, n_samples, 1)
        normal_sigma2 = sigma2 / x["nu"].unsqueeze(2)
        x = torch.cat((normal_mu, normal_sigma2), dim=2)
        return x
