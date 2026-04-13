"""Shared credal net implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.method.method import predictor_transformation
from probly.predictor import Predictor, ProbabilisticClassifier, RepresentationPredictor
from probly.representation.distribution import CategoricalDistribution
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, lazydispatch_traverser, traverse


@runtime_checkable
class CredalNetPredictor[**In, Out: CategoricalDistribution](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the Credal Bayesian Neural Network transformation."""


REPLACED = GlobalVariable[bool]("REPLACED", default=False)

credal_net_traverser = lazydispatch_traverser[object](name="credal_net_traverser")


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier,),
    preserve_predictor_type=False,
)  # ty:ignore[invalid-argument-type]
@CredalNetPredictor.register_factory
def credal_net[**In, Out: CategoricalDistribution](base: Predictor[In, Out]) -> CredalNetPredictor[In, Out]:
    """Create a credal net predictor from a base predictor based on :cite:`wang2024credalnet`.

    Args:
        base: Predictor, The base model to be used for credal net.
        num_classes: int, The number of classes to predict.

    Returns:
        Predictor, The credal net predictor.
    """
    return traverse(base, nn_compose(credal_net_traverser), init={REPLACED: False, TRAVERSE_REVERSED: True})
