"""Shared credal net implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from probly.method.method import predictor_transformation
from probly.predictor import Predictor, ProbabilisticClassifier, RepresentationPredictor
from probly.representation.distribution import CategoricalDistribution


@runtime_checkable
class CredalNetPredictor[**In, Out: CategoricalDistribution](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the Credal Bayesian Neural Network transformation."""


@lazydispatch
def credal_net_generator[**In, Out: CategoricalDistribution](base: Predictor[In, Out]) -> CredalNetPredictor[In, Out]:
    """Generate a credal net from a base model."""
    msg = f"No credal net generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


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
    return credal_net_generator(base)
