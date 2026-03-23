"""Module for predictors."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from .common import EnsemblePredictor, Predictor, predict


@predict.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["EnsemblePredictor", "Predictor", "predict"]
