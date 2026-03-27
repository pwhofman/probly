"""Shared credal wrapper implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.method.ensemble import EnsemblePredictor
from probly.method.ensemble._common import ensemble

if TYPE_CHECKING:
    from probly.predictor import Predictor


class CredalWrapperPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor that applies the credal wrapper representer."""


@CredalWrapperPredictor.register_factory
def credal_wrapper[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> EnsemblePredictor[In, Out]:
    """Create a credal wrapper predictor from a base predictor based on :cite:`wangCredalWrapper2024`.

    Args:
        base: Predictor, The base model to be used for the credal wrapper ensemble.
        num_members: The number of members in the credal wrapper ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        Predictor, The credal wrapper ensemble predictor.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)
