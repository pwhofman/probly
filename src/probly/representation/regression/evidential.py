"""Evidential model class implementation."""

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

    # TODO: Implement sample method
