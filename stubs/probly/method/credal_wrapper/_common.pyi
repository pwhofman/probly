"""Shared credal wrapper implementation."""

from __future__ import annotations
import probly
from typing import Literal

from typing import TYPE_CHECKING, Protocol

from probly.method.ensemble import EnsemblePredictor
from probly.method.ensemble._common import ensemble
from probly.method.method import predictor_transformation
from probly.predictor import ProbabilisticClassifier

if TYPE_CHECKING:
    from probly.predictor import Predictor


class CredalWrapperPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor that applies the credal wrapper representer."""
def credal_wrapper[**In, Out](base: Predictor[In, Out], num_members: int, reset_params: bool = True, *, predictor_type: Literal['categorical_distribution_predictor', 'probabilistic_classifier'] | type[probly.predictor.CategoricalDistributionPredictor] | None = None) -> EnsemblePredictor[In, Out]:
    """Create a credal wrapper predictor from a base predictor based on :cite:`wangCredalWrapper2024`.

    Args:
        base: Predictor, The base model to be used for the credal wrapper ensemble.
        num_members: The number of members in the credal wrapper ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        Predictor, The credal wrapper ensemble predictor.
    """
    ...
