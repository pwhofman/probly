"""Dropout class implementation."""

import copy

import torch
from torch import nn


class Dropout(nn.Module):
    """Implementation of a Dropout ensemble class to be used for uncertainty quantification.

    Attributes:
        p: float, The probability of dropout.
        model: torch.nn.Module, The model with Dropout layers.

    """

    def __init__(self, base: nn.Module, p: float = 0.25) -> None:
        """Initialize an instance of the Dropout class.

        Args:
            base: torch.nn.Module, The base model to be used for dropout.
            p: float, The probability of dropping out a neuron.
        """
        super().__init__()
        self.p = p
        self._convert(base)

    def forward(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Forward pass of the dropout ensemble.

        Args:
            x: torch.Tensor, input data
            n_samples: int, number of samples
        Returns:
            torch.Tensor, ensemble output

        """
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)

    def _convert(self, base: nn.Module) -> None:
        """Convert base model to a dropout model.

        Convert base model by looping through all the layers
        and adding a dropout layer before each linear layer.

        Args:
            base: torch.nn.Module, The base model to be used for dropout.

        """
        self.model = copy.deepcopy(base)

        def apply_dropout(module: nn.Module, *, first_layer: bool = True) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear) and not first_layer:
                    setattr(module, name, nn.Sequential(nn.Dropout(p=self.p), child))  # add dropout
                else:
                    if first_layer and not isinstance(child, nn.Sequential):  # ignore Sequential layers as first layers
                        first_layer = False  # skip first layer
                    apply_dropout(child, first_layer=first_layer)  # apply recursively to all layers

        apply_dropout(self.model)

    def eval(self) -> None:
        """Sets the model to evaluation mode, but keeps the dropout layers active."""
        super().eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
