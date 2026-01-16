"""Torch dropout implementation."""

from __future__ import annotations

from typing import Any

from torch import nn

from .common import register


def prepend_torch_dropout(obj: nn.Module, p: float, rngs: Any) -> nn.Sequential:  # noqa: ANN401, ARG001
    """Prepend a Dropout layer before the given layer based on :cite:`galDropoutBayesian2016`.

    This construction allows for Monte Carlo Dropout inference by keeping the dropout layer active during prediction.
    """
    return nn.Sequential(nn.Dropout(p=p), obj)


register(nn.Linear, prepend_torch_dropout)
