"""Shared credal relative likelihood implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.method.ensemble import EnsemblePredictor
from probly.method.ensemble._common import ensemble
from probly.method.method import predictor_transformation
from probly.predictor import ProbabilisticClassifier

if TYPE_CHECKING:
    from probly.predictor import Predictor


class CredalRelativeLikelihoodPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor that applies the credal relative likelihood transformation."""


@predictor_transformation(permitted_predictor_types=(ProbabilisticClassifier,))
@CredalRelativeLikelihoodPredictor.register_factory
def credal_relative_likelihood[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> CredalRelativeLikelihoodPredictor[In, Out]:
    """Create a credal relative likelihood predictor from a base predictor based on :cite:`lohr2025credal`.

    Args:
        base: The base model to be used for the credal relative likelihood ensemble.
        num_members: The number of members in the credal relative likelihood ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        The credal relative likelihood ensemble predictor.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)
