"""Predictor for torch models."""

from __future__ import annotations

from typing import Any

from torch import nn

from .common import predict


@predict.register(nn.ModuleList)
def _(model: nn.ModuleList, *args: Any, **kwargs: Any) -> list:  # noqa: ANN401
    """Predicts using a torch ensemble."""
    return [member(*args, **kwargs) for member in model]
