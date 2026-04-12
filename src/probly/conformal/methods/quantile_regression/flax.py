"""Conformalise a Flax model for quantile regression."""

from __future__ import annotations

from flax import nnx

from ._common import conformal_generator


@conformal_generator.register(nnx.Module)
def _(
    model: nnx.Module,
) -> nnx.Module:
    """Conformalise a Flax model."""
    model.conformal_quantile = None
    model.non_conformity_score = None
    return model