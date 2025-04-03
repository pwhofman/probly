"""Evidental deep learning model class implementation."""

import copy

import torch
from torch import nn


class Evidential(nn.Module):
    """Implementation of an evidential deep learning model to be used for uncertainty quantification.

    Attributes:
        model: torch.nn.Module, the model with an activation function suitable
        for evidential classification.

    """

    def __init__(self, base: nn.Module, activation: nn.Module = nn.Softplus()) -> None:  # noqa: B008
        """Initialize an evidential model by converting the base model into an evidential model.

        Args:
            base: torch.nn.Module, the base model to be used
            activation: torch.nn.Module, the activation function that will be used
        """
        super().__init__()
        self._convert(base, activation)

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

    def _convert(self, base: nn.Module, activation: nn.Module) -> None:
        """Convert a model into an evidential deep learning model.

        Args:
            base: torch.nn.Module, the base model to be used.
            activation: torch.nn.Module, the activation function that will be used.

        """
        self.model = nn.Sequential(copy.deepcopy(base), activation)

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample from the predicted distribution for a given input x.

        Args:
            x: torch.Tensor, input data
            n_samples: int, number of samples
        Returns:
            torch.Tensor, samples

        """
        dirichlet = torch.distributions.Dirichlet(self.model(x) + 1.0)
        return torch.stack([dirichlet.sample() for _ in range(n_samples)]).swapaxes(0, 1)
