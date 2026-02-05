"""Base module with the purpose of codesharing for platt, vector and temperature scaling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from probly.utils.torch import torch_collect_outputs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.utils.data import DataLoader


class ScalerTorch(nn.Module, ABC):
    """Base class for Torch Scaling Implementations."""

    def __init__(self, base: nn.Module, num_classes: int) -> None:
        """Initialize the scaling class with base model and number of classes.

        Args:
            base: The base model that should be calibrated.
            num_classes: The number of classes the base model was trained on.
        """
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        self.device = next(base.parameters()).device

        # freeze base model
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.base.eval()

    def forward(self, x: Tensor) -> Tensor:
        """Produce scaled logits based on the input.

        Args:
            x: The input used to generate logits.

        Returns:
            scaled_logits: The scaled logits based on the input.
        """
        x = x.to(self.device)
        logits = cast("Tensor", self.base(x))
        return self._scale_logits(logits)

    @abstractmethod
    def _scale_logits(self, logits: Tensor) -> Tensor:
        """Scale logits base on learned parameters."""
        raise NotImplementedError

    @abstractmethod
    def _parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        """Create an iterable of all parameters to be optimized."""
        raise NotImplementedError

    @abstractmethod
    def _loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Calibration loss function."""
        raise NotImplementedError

    def fit(self, calibration_set: DataLoader, learning_rate: float = 0.01, max_iter: int = 50) -> None:
        """Optimize the parameters on a calibration set.

        Args:
            calibration_set: The dataset used for the optimization of the parameters.
            learning_rate: The learning rate the optimizer uses.
            max_iter: The maximum ammount of iterations / steps the optimizer can take.
        """
        logits, labels = torch_collect_outputs(self.base, calibration_set, self.device)

        optimizer = torch.optim.LBFGS(
            list(self._parameters_to_optimize()),
            lr=learning_rate,
            max_iter=max_iter,
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = self._loss_fn(self._scale_logits(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

    def predict(self, x: Tensor) -> Tensor:
        """Make calibrated predictions based on the input.

        Args:
            x: The input on which to make predictions on.

        Returns:
            probs: The calibrated prediction of the model.
        """
        self.eval()
        with torch.no_grad():
            scaled_logits = self.forward(x)

            if self.num_classes == 1:
                return torch.sigmoid(scaled_logits)

            return torch.softmax(scaled_logits, dim=1)
