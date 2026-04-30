"""Shared dare implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.method.ensemble import EnsemblePredictor, ensemble
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class DarePredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """Protocol for dare predictors."""


@predictor_transformation(
    permitted_predictor_types=None,
)
@DarePredictor.register_factory(autocast_builtins=True)
def dare[**In, Out](base: Predictor[In, Out], num_members: int, reset_params: bool = True) -> DarePredictor[In, Out]:
    """Create a dare predictor from a base predictor based on :cite:`arXiv:2304.04042`.

    Args:
        base: Predictor, the base model to be used for dare.
        num_members: The number of members to generate.
        reset_params: Whether to reset the predictor parameters.

    Returns:
        Predictor, The dare predictor.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)
