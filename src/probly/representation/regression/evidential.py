import copy

import torch
import torch.nn as nn

from ..layers import NormalInverseGammaLinear


class Evidential(nn.Module):
    """
    This class implements an evidential deep learning model for regression.
    Args:
        base: torch.nn.Module, The base model to be used.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self._convert(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output
        """
        return self.model(x)

    def _convert(self, base: nn.Module) -> None:
        """
        Convert a model into an evidential deep learning regression model.
        Replaces the last layer by a layer parameterizing a normal inverse gamma distribution.
        Args:
            base: torch.nn.Module, The base model to be used.
        """
        self.model = copy.deepcopy(base)
        for name, child in reversed(list(self.model.named_children())):
            if isinstance(child, nn.Linear):
                setattr(self.model, name,
                        NormalInverseGammaLinear(child.in_features, child.out_features))
                break

    #TODO: Implement sample method
