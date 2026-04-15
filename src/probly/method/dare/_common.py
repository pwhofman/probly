"""Shared dare implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from probly.method.ensemble import EnsemblePredictor
from probly.method.method import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class DarePredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """Protocol for dare predictors."""


@lazydispatch
def dare_generator[**In, Out](
    base: Predictor[In, Out],
    num_members: int,
    lambda_reg: float,
    threshold: float,
) -> DarePredictor[In, Out]:
    """Generate a dare from a base model."""
    msg = f"No dare generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(
    permitted_predictor_types=None,
)
@DarePredictor.register_factory
def dare[**In, Out](
    base: Predictor[In, Out],
    num_members: int = 1,
    lambda_reg: float = 0.01,
    threshold: float = 0.1,
) -> DarePredictor[In, Out]:
    """Create a dare predictor from a base predictor based on :cite:`arXiv:2304.04042`.

    Args:
        base: Predictor, the base model to be used for dare.
        num_members: The number of members to generate.
        lambda_reg: The anti-regularization coefficient.
        threshold: The loss threshold to trigger anti-regularization.

    Returns:
        Predictor, The dare predictor.
    """
    return dare_generator(base, num_members=num_members, lambda_reg=lambda_reg, threshold=threshold)
