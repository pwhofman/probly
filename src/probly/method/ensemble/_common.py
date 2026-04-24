"""Shared ensemble implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from flextype import flexdispatch
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
        if isinstance(instance, list | tuple) and all(isinstance(p, Predictor) for p in instance):
            return True
        return NotImplemented


@runtime_checkable
class EnsembleCategoricalDistributionPredictor[**In, Out: CategoricalDistribution](
    EnsemblePredictor[In, Out], Protocol
):
    """Protocol for ensemble predictors that return a categorical distribution."""

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        if isinstance(instance, list | tuple) and all(
            isinstance(p, CategoricalDistributionPredictor) for p in instance
        ):
            return True
        return NotImplemented


@runtime_checkable
class EnsembleDirichletDistributionPredictor[**In, Out: DirichletDistribution](EnsemblePredictor[In, Out], Protocol):
    """Protocol for ensemble predictors that return a categorical distribution."""

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        if isinstance(instance, Iterable) and all(isinstance(p, DirichletDistributionPredictor) for p in instance):
            return True
        return NotImplemented


@flexdispatch
def ensemble_generator[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> EnsemblePredictor[In, Out]:
    """Generate an ensemble from a base model."""
    msg = f"No ensemble generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


def register_ensemble_members(ensemble: EnsemblePredictor, t: type[Predictor] | None) -> EnsemblePredictor:
    """Register the members of an ensemble predictor."""
    if t is None:
        return ensemble
    for member in ensemble:
        t.register_instance(member)

    return ensemble


@predictor_transformation(
    permitted_predictor_types=None,
    post_transform=register_ensemble_members,
)
@EnsemblePredictor.register_factory
def ensemble[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> EnsemblePredictor[In, Out]:
    """Create an ensemble predictor from a base predictor based on :cite:`lakshminarayananSimpleScalable2017`.

    Args:
        base: Predictor, The base model to be used for the ensemble.
        num_members: The number of members in the ensemble.
        reset_params: Whether to reset the parameters of each member.

    Returns:
        Predictor, The ensemble predictor.
    """
    return ensemble_generator(base, num_members=num_members, reset_params=reset_params)


@predict_raw.register(EnsemblePredictor)
def predict_list[**In, Out](predictor: EnsemblePredictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> list[Out]:
    """Predict for a list of predictors."""
    return [predict(p, *args, **kwargs) for p in predictor]
