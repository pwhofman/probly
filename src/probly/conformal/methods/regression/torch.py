"""Conformal regression methods for PyTorch models."""

from __future__ import annotations

from torch import nn

from ._common import conformal_generator


@conformal_generator.register(nn.Module)
def _(model: nn.Module) -> nn.Module:
    """Conformalise a PyTorch model."""
    model.conformal_quantile = None  # ty: ignore[unresolved-attribute]
    model.non_conformity_score = None  # ty: ignore[unresolved-attribute]
    return model
