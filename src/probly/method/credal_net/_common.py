"""Shared credal net implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.method.method import predictor_transformation
from probly.predictor import Predictor, ProbabilisticClassifier, RepresentationPredictor, predict, predict_raw
from probly.representation.credal_set import (
    ProbabilityIntervalsCredalSet,
    create_probability_intervals_from_lower_upper_array,
)
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse


@runtime_checkable
class CredalNetPredictor[**In, Out: ProbabilityIntervalsCredalSet](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the Credal Bayesian Neural Network transformation."""


REPLACED = GlobalVariable[bool]("REPLACED", default=False)

credal_net_traverser = flexdispatch_traverser[object](name="credal_net_traverser")


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier,),
    preserve_predictor_type=False,
)  # ty:ignore[invalid-argument-type]
@CredalNetPredictor.register_factory
def credal_net[**In, Out: ProbabilityIntervalsCredalSet](base: Predictor[In, Out]) -> CredalNetPredictor[In, Out]:
    """Create a credal net predictor from a base predictor based on :cite:`wang2024credalnet`.

    Args:
        base: Predictor, The base model to be used for credal net.
        num_classes: int, The number of classes to predict.

    Returns:
        Predictor, The credal net predictor.
    """
    return traverse(base, nn_compose(credal_net_traverser), init={REPLACED: False, TRAVERSE_REVERSED: True})


@predict.register(CredalNetPredictor)
def _[**In](
    predictor: CredalNetPredictor[In, ProbabilityIntervalsCredalSet], *args: In.args, **kwargs: In.kwargs
) -> ProbabilityIntervalsCredalSet:
    """Predict with a credal net predictor."""
    return create_probability_intervals_from_lower_upper_array(predict_raw(predictor, *args, **kwargs))
