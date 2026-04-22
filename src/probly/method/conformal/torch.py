"""PyTorch conformal predictor wrappers."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ._common import (
    _ConformalPredictorBase,
    conformal_generator,
)

if TYPE_CHECKING:
    from probly.conformal_scores import NonConformityScore


@conformal_generator.register(nn.Module)
class TorchConformalSetPredictor[**In, Out](_ConformalPredictorBase[In, Out], nn.Module, ABC):
    """Base torch conformal wrapper forwarding ``forward``."""

    predictor: nn.Module

    def __init__(self, predictor: nn.Module, non_conformity_score: NonConformityScore[Out, torch.Tensor]) -> None:
        """Initialize the torch conformal wrapper."""
        super().__init__(predictor, non_conformity_score)
        self.register_buffer("_conformal_quantile", torch.tensor(float("nan"), dtype=torch.float64))

    def forward(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Forward to the wrapped model."""
        return self.predictor(*args, **kwargs)

    def _quantile_buffer(self) -> torch.Tensor:
        """Return the persistent conformal quantile buffer."""
        return cast("torch.Tensor", self._conformal_quantile)

    def calibrate(self, alpha: float, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Any:  # noqa: ANN401
        """Calibrate and persist quantile state as a torch buffer."""
        calibrated = super().calibrate(alpha, y_calib, *calib_args, **calib_kwargs)
        quantile_buffer = self._quantile_buffer()
        if self.conformal_quantile is None:
            quantile_buffer.fill_(float("nan"))
        else:
            quantile_buffer.fill_(float(self.conformal_quantile))
        return calibrated

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load state and restore the scalar conformal quantile attribute."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        quantile_buffer = self._quantile_buffer()
        if torch.isnan(quantile_buffer).item():
            self.conformal_quantile = None
        else:
            self.conformal_quantile = float(quantile_buffer.item())
