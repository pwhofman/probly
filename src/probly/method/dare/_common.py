"""Shared dare implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from probly.method.method import predictor_transformation
from probly.predictor import (
    Predictor,
    RandomPredictor,
)


@runtime_checkable
class DarePredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """Protocol for dare predictors."""


@lazydispatch
def dare_generator[**In, Out](base: Predictor[In, Out], delta: float, num_members: int) -> Predictor[In, Out]:
    """Generate a dare from a base model."""
    msg = f"No dare generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(
    permitted_predictor_types=None,
)  # ty: ignore[invalid-argument-type]
@DarePredictor.register_factory
def dare[**In, Out](
    base: Predictor[In, Out],
    delta: float = 0.05,
    num_members: int = 1,
) -> DarePredictor[In, Out]:
    """Create a dare predictor from a base predictor based on :cite:`arXiv:2304.04042`.

    Args:
        base: The base model to be wrapped with DARE.
        delta: float, the distance between the members.
        num_members: int, The number of members to generate.

    Returns:
        The dare model.
    """
    return dare_generator(base, num_members=num_members, delta=delta)
