"""Shared credal relative likelihood implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.transformation.ensemble.common import ensemble

if TYPE_CHECKING:
    from probly.predictor import EnsemblePredictor, Predictor


def credal_relative_likelihood[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> EnsemblePredictor[In, Out]:
    """Create a credal relative likelihood predictor from a base predictor based on :cite:`lohr2025credal`.

    Args:
        base: Predictor, The base model to be used for the credal relative likelihood ensemble.
        num_members: The number of members in the credal relative likelihood ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        EnsemblePredictor, The credal relative likelihood ensemble predictor.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)
