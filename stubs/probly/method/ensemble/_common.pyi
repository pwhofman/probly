"""Shared ensemble implementation."""

from __future__ import annotations
import probly

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from probly.method.method import predictor_transformation
from probly.predictor import (
    CategoricalDistributionPredictor,
    DirichletDistributionPredictor,
    IterablePredictor,
    Predictor,
    predict,
    predict_raw,
)
from probly.representation.distribution import CategoricalDistribution, DirichletDistribution


@runtime_checkable
class EnsemblePredictor[**In, Out](IterablePredictor[In, Out], Iterable[Predictor[In, Out]], Protocol):
    """Protocol for ensemble predictors."""

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        ...


@runtime_checkable
class EnsembleCategoricalDistributionPredictor[**In, Out: CategoricalDistribution](
    EnsemblePredictor[In, Out], Protocol
):
    """Protocol for ensemble predictors that return a categorical distribution."""

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        ...


@runtime_checkable
class EnsembleDirichletDistributionPredictor[**In, Out: DirichletDistribution](EnsemblePredictor[In, Out], Protocol):
    """Protocol for ensemble predictors that return a categorical distribution."""

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        ...


@lazydispatch
def ensemble_generator[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> EnsemblePredictor[In, Out]:
    """Generate an ensemble from a base model."""
    ...


def register_ensemble_members(ensemble: EnsemblePredictor, t: type[Predictor] | None) -> EnsemblePredictor:
    """Register the members of an ensemble predictor."""
    ...
def ensemble[**In, Out](base: Predictor[In, Out], num_members: int, reset_params: bool = True, *, predictor_type: probly.predictor.PredictorName | type[probly.predictor.Predictor] | None = None) -> EnsemblePredictor[In, Out]:
    """Create an ensemble predictor from a base predictor based on :cite:`lakshminarayananSimpleScalable2017`.

    Args:
        base: Predictor, The base model to be used for the ensemble.
        num_members: The number of members in the ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        Predictor, The ensemble predictor.
    """
    ...


@predict_raw.register(EnsemblePredictor)
def predict_list[**In, Out](predictor: EnsemblePredictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> list[Out]:
    """Predict for a list of predictors."""
    ...
