"""Shared credal ensembling implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.method.ensemble import EnsemblePredictor
from probly.method.ensemble._common import ensemble

if TYPE_CHECKING:
    from probly.predictor import Predictor


class CredalEnsemblingPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor that applies the credal ensembling representer."""


@CredalEnsemblingPredictor.register_factory
def credal_ensembling[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> CredalEnsemblingPredictor[In, Out]:
    """Create a credal ensembling predictor from a base predictor based on :cite:`nguyenCredalEnsembling2025`.

    Args:
        base: Predictor, The base model to be used for the credal ensembling ensemble.
        num_members: The number of members in the credal ensembling ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        EnsemblePredictor, The credal ensembling ensemble predictor.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)
