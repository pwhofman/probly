from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch
from torch import nn

from ._common import (
    _ConformalCredalSetPredictorBase,
    conformal_credal_set_generator
)

if TYPE_CHECKING:
    from probly.conformal_scores._common import NonConformityScore

@conformal_credal_set_generator.register(nn.Module)
class TorchConformalCredalSetPredictor[**In, Out](_ConformalCredalSetPredictorBase[In, Out], nn.Module, ABC):
    """Base torch conformal wrapper."""

    predictor: nn.Module

    def __init__(self, predictor: nn.Module, non_conformity_score: NonConformityScore[Out, torch.Tensor]) -> None:
        super().__init__(predictor, non_conformity_score)
        self.register_buffer("_conformal_quantile", torch.tensor(float("nan"), dtype=torch.float64))

    @property
    def conformal_quantile(self) -> float | None:
        """Return the calibrated conformal quantile if available."""
        quantile = self._buffers.get("_conformal_quantile")
        if not isinstance(quantile, torch.Tensor) or torch.isnan(quantile).item():
            return None
        return float(quantile.item())

    @conformal_quantile.setter
    def conformal_quantile(self, value: float | None) -> None:
        """Persist conformal quantile in a torch buffer for serialization."""
        quantile = self._buffers.get("_conformal_quantile")
        if not isinstance(quantile, torch.Tensor):
            return
        quantile.fill_(float("nan") if value is None else float(value))

    def forward(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Forward to the wrapped model."""
        return self.predictor(*args, **kwargs)
