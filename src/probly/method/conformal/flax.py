"""Flax conformal predictor wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flax import nnx

from ._common import (
    _ConformalPredictorBase,
    conformal_generator,
)

if TYPE_CHECKING:
    from jax import Array

    from probly.conformal_scores._common import NonConformityScore


@conformal_generator.register(nnx.Module)
class FlaxConformalSetPredictor[**In, Out](_ConformalPredictorBase[In, Out], nnx.Module):
    """Base flax conformal wrapper forwarding ``__call__``."""

    predictor: nnx.Module

    def __init__(self, predictor: nnx.Module, non_conformity_score: NonConformityScore[Out, Array]) -> None:
        """Initialize the flax conformal wrapper."""
        super().__init__(predictor, non_conformity_score)

    def __call__(self, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        """Forward to the wrapped model."""
        return self.predictor(*args, **kwargs)
