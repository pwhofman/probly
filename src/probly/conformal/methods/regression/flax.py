"""Conformal regression methods for JAX/Flax models."""

from __future__ import annotations

from flax import nnx

from probly.layers.flax import ConformalRegressionHead

from ._common import conformal_generator


@conformal_generator.register(nnx.Module)
def _(model: nnx.Module) -> nnx.Module:
    """Conformalise a Flax model."""
    model.non_conformity_score = None
    model.conformal_quantile = None
    return model
