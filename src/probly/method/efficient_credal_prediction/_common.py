"""Shared efficient credal prediction implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from probly.method.method import predictor_transformation
from probly.predictor import LogitClassifier, RepresentationPredictor
from probly.representation.distribution import CategoricalDistribution

if TYPE_CHECKING:
    from collections.abc import Iterable

    from probly.predictor import Predictor


@runtime_checkable
class EfficientCredalPredictor[**In, Out: CategoricalDistribution](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the efficient credal prediction method."""

    predictor: Predictor
    lower: Iterable[float]
    upper: Iterable[float]

    @property
    def lower_bounds(self) -> Iterable[float]:
        return self.lower

    @property
    def upper_bounds(self) -> Iterable[float]:
        return self.upper


@lazydispatch
def efficient_credal_prediction_generator[**In, Out: CategoricalDistribution](
    base: Predictor,
    num_classes: int,
) -> EfficientCredalPredictor[In, Out]:
    """Generate an efficient credal predictor from a base model."""
    msg = f"No efficient credal prediction generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@EfficientCredalPredictor.register_factory
def efficient_credal_prediction[**In, Out: CategoricalDistribution](
    base: Predictor,
    num_classes: int,
) -> EfficientCredalPredictor[In, Out]:
    """Create an efficient credal predictor from a base predictor based on :cite:`hofmanefficient`.

    Args:
        base: Predictor, The base model to be used for the efficient credal predictor.
        num_classes: int, The number of classes to predict.

    Returns:
        Predictor, The efficient credal predictor.
    """
    return efficient_credal_prediction_generator(base, num_classes)
