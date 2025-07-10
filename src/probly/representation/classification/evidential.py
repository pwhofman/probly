"""Evidential deep learning model class implementation."""

from __future__ import annotations

import copy
from typing import Unpack

import torch
from torch import nn

from probly.representation.predictor import RepresentationPredictor


class Evidential[In, KwIn](
    nn.Module,
    RepresentationPredictor[In, KwIn, torch.Tensor, torch.Tensor],
):
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

    def forward(self, *args: In, **kwargs: Unpack[KwIn]) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            *args: Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Model output.

        """
        return self.model(*args, **kwargs)

    def predict_pointwise(self, *args: In, **kwargs: Unpack[KwIn]) -> torch.Tensor:
        """Forward pass of the model for point-wise prediction.

        Args:
            *args: Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Model output.

        """
        alphas = self.model(*args, **kwargs) + 1.0
        return alphas / alphas.sum(dim=1, keepdim=True)

    def predict_representation(self, *args: In, **kwargs: Unpack[KwIn]) -> torch.Tensor:
        """Forward pass of the model for uncertainty representation.

        Args:
            *args: Input data.
            **kwargs: Additional keyword arguments.

        Returns:
            Model output.


        """
        return self.model(*args, **kwargs)

    def _convert(self, base: nn.Module, activation: nn.Module) -> None:
        """Convert a model into an evidential deep learning model.

        Args:
            base: torch.nn.Module, the base model to be used.
            activation: torch.nn.Module, the activation function that will be used.

        """
        self.model = nn.Sequential(copy.deepcopy(base), activation)

    def sample(self, *args: In, n_samples: int, **kwargs: Unpack[KwIn]) -> torch.Tensor:
        """Sample from the predicted distribution for a given input x.

        Args:
            *args: Input data.
            n_samples: Number of samples.
            **kwargs: Additional keyword arguments.

        Returns:
            Samples from the Dirichlet distribution.

        """
        dirichlet = torch.distributions.Dirichlet(self.model(*args, **kwargs) + 1.0)
        return torch.stack([dirichlet.sample() for _ in range(n_samples)]).swapaxes(0, 1)
