"""Credal wrapper method compatibility layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.predictor import LogitClassifier, ProbabilisticClassifier
from probly.representer import ProbabilityIntervalsRepresenter, representer
from probly.transformation.ensemble import EnsemblePredictor, ensemble
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


class CredalWrapperPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the credal wrapper representer."""


@predictor_transformation(
    permitted_predictor_types=((ProbabilisticClassifier, LogitClassifier)), preserve_predictor_type=False
)
@CredalWrapperPredictor.register_factory(autocast_builtins=True)
def credal_wrapper[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> CredalWrapperPredictor[In, Out]:
    """Create a credal wrapper predictor from a base predictor based on :cite:`wangCredalWrapper2024`.

    Args:
        base: Predictor, The base classifier to replicate into an ensemble.
        num_members: int, Number of ensemble members.
        reset_params: bool, Whether to reset the parameters of each member. Default is True.

    Returns:
        CredalWrapperPredictor, The credal wrapper predictor outputting a ProbabilityIntervalsCredalSet.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)


representer.register(CredalWrapperPredictor, ProbabilityIntervalsRepresenter)

__all__ = ["CredalWrapperPredictor", "credal_wrapper"]
