"""Dropout/DropConnect representation for uncertainty quantification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from probly.representation.predictor import RepresentationPredictorWrapper
from probly.traverse import CLONE, GlobalVariable, Traverser
from probly.traverse_nn import nn_traverse

P = GlobalVariable[float]("P", "The probability of dropout.", default=0.25)


class Drop(nn.Module, RepresentationPredictorWrapper):
    """This class implements a generic drop layer to be used for uncertainty quantification."""

    _convert_traverser: Traverser[nn.Module]
    _eval_traverser: Traverser[nn.Module]

    def __init__(
        self,
        base: nn.Module,
        p: float = P.default,
    ) -> None:
        super().__init__()
        self.p = p
        self.model = self._convert(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the dropout model.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, ensemble output

        """
        return self.model(x)

    def predict_pointwise(
        self,
        x: torch.Tensor,
        n_samples: int,
        logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass that gives a point-wise prediction by taking the mean over the samples.

        Args:
            x: torch.Tensor, input data
            n_samples: int, number of samples
            logits: bool, whether to return logits or probabilities
        Returns:
            torch.Tensor, point-wise prediction
        """
        if logits:
            return torch.stack([self.model(x) for _ in range(n_samples)], dim=1).mean(
                dim=1,
            )
        return torch.stack(
            [F.softmax(self.model(x), dim=1) for _ in range(n_samples)],
            dim=1,
        ).mean(dim=1)

    def predict_representation(
        self,
        x: torch.Tensor,
        n_samples: int,
        logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass that gives an uncertainty representation.

        Args:
            x: torch.Tensor, input data
            n_samples: int, number of samples
            logits: bool, whether to return logits or probabilities
        Returns:
            torch.Tensor, uncertainty representation
        """
        if logits:
            return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)
        return torch.stack(
            [F.softmax(self.model(x), dim=1) for _ in range(n_samples)],
            dim=1,
        )

    def _convert(self, base: nn.Module) -> nn.Module:
        """Convert base model to a dropout model.

        Convert base model by looping through all the layers
        and adding a dropout layer before each linear layer.

        Args:
            base: torch.nn.Module, The base model to be used for drop.
        """
        return nn_traverse(base, self._convert_traverser, init={P: self.p, CLONE: True})

    def eval(self) -> Drop:
        """Sets the model to evaluation mode, but keeps the drop layers active."""
        super().eval()

        nn_traverse(self.model, self._eval_traverser, init={CLONE: False})

        return self
