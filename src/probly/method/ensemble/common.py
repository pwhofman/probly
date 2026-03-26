"""Shared ensemble implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from lazy_dispatch import lazydispatch
from probly.lazy_types import FLAX_LIST, TORCH_MODULE_LIST
from probly.predictor import Predictor, predict
from probly.predictor.common import IterablePredictor


class EnsemblePredictor[**In, Out](IterablePredictor[In, Out], Iterable[Predictor[In, Out]], Protocol):
    """Protocol for ensemble predictors."""


EnsemblePredictor.register(
    (
        list,
        FLAX_LIST,
        TORCH_MODULE_LIST,
    )
)


@lazydispatch
def ensemble_generator[**In, Out](base: Predictor[In, Out]) -> EnsemblePredictor[In, Out]:
    """Generate an ensemble from a base model."""
    msg = f"No ensemble generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


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


@predict.register(EnsemblePredictor)
def predict_list[**In, Out](predictor: EnsemblePredictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> list[Out]:
    """Predict for a list of predictors."""
    return [predict(p, *args, **kwargs) for p in predictor]
