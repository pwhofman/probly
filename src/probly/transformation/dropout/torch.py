"""Torch dropout implementation."""

from __future__ import annotations

from torch import nn

from .common import register


def prepend_torch_dropout(obj: nn.Module, p: float) -> nn.Sequential:
    """Prepend a Dropout layer before the given layer.

    This construction allows for Monte Carlo Dropout inference by keeping the dropout layer active during prediction.

    References:
        Based on 'Dropout as a Bayesian Approximation' by Y. Gal and Z. Ghahramani (2016).
        See: :cite:`gal2016dropout`

    """
    return nn.Sequential(nn.Dropout(p=p), obj)


register(nn.Linear, prepend_torch_dropout)
