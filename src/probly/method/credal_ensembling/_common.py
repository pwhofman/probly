"""Shared credal ensembling implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.method.ensemble import EnsembleCategoricalDistributionPredictor
from probly.method.ensemble._common import ensemble
from probly.method.method import predictor_transformation
from probly.predictor import LogitClassifier, ProbabilisticClassifier
from probly.representation.distribution import CategoricalDistribution

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class CredalEnsemblingPredictor[**In, Out: CategoricalDistribution](
    EnsembleCategoricalDistributionPredictor[In, Out], Protocol
):
    """A predictor that applies the credal ensembling representer."""


@predictor_transformation(
    permitted_predictor_types=(
        ProbabilisticClassifier,
        LogitClassifier,
    ),
    preserve_predictor_type=False,
)
@CredalEnsemblingPredictor.register_factory
def credal_ensembling[**In, Out: CategoricalDistribution](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> CredalEnsemblingPredictor[In, Out]:
    """Create a credal ensembling predictor from a base predictor based on :cite:`nguyenCredalEnsembling2025`.

    Args:
        base: The base model to be used for the credal ensembling ensemble.
        num_members: The number of members in the credal ensembling ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        The credal ensembling ensemble predictor.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)
