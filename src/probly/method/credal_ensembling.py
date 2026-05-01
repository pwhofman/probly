"""Credal ensembling method compatibility layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from probly.predictor import LogitClassifier, ProbabilisticClassifier
from probly.representation.distribution import CategoricalDistribution
from probly.representer import RepresentativeConvexCredalSetRepresenter, representer
from probly.transformation.ensemble import EnsembleCategoricalDistributionPredictor, ensemble
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class CredalEnsemblingPredictor[**In, Out: CategoricalDistribution](
    EnsembleCategoricalDistributionPredictor[In, Out], Protocol
):
    """A predictor routed through the credal ensembling representer."""


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier),
    preserve_predictor_type=False,
)
@CredalEnsemblingPredictor.register_factory(autocast_builtins=True)
def credal_ensembling[**In, Out: CategoricalDistribution](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> CredalEnsemblingPredictor[In, Out]:
    """Create a credal ensembling predictor from a base predictor."""
    return ensemble(base, num_members=num_members, reset_params=reset_params)


representer.register(CredalEnsemblingPredictor, RepresentativeConvexCredalSetRepresenter)


__all__ = ["CredalEnsemblingPredictor", "credal_ensembling"]
