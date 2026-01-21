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


class _LogitScaler(nn.Module, ABC):
    """Base class for logit calibration methods."""

    def __init__(self, base: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        self.device = next(base.parameters()).device

        # freeze base model
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.base.eval()

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        logits = cast("Tensor", self.base(x))
        return self._scale_logits(logits)

    @abstractmethod
    def _scale_logits(self, logits: Tensor) -> Tensor:
        """Apply calibration transform."""
        raise NotImplementedError

    @abstractmethod
    def _parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        """Return trainable parameters."""
        raise NotImplementedError

    @abstractmethod
    def _loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Calibration loss."""
        raise NotImplementedError

    def fit(self, calibration_set: DataLoader, learning_rate: float = 0.01, max_iter: int = 50) -> None:
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
        self.eval()
        with torch.no_grad():
            scaled_logits = self.forward(x)

            if self.num_classes == 1:
                return torch.sigmoid(scaled_logits)

            return torch.softmax(scaled_logits, dim=1)
