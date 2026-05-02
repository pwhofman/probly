"""DARE method compatibility layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.transformation.ensemble import EnsemblePredictor, ensemble
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class DarePredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the DARE method API."""


@predictor_transformation(permitted_predictor_types=None)
@DarePredictor.register_factory(autocast_builtins=True)
def dare[**In, Out](base: Predictor[In, Out], num_members: int, reset_params: bool = True) -> DarePredictor[In, Out]:
    """Create a DARE predictor from a base predictor."""
    return ensemble(base, num_members=num_members, reset_params=reset_params)


__all__ = ["DarePredictor", "dare"]
