"""Shared Dirichlet clipped-exp + 1 activation transformation implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.predictor import EvidentialPredictor, LogitClassifier, predict, predict_raw
from probly.representation.distribution import DirichletDistribution, create_dirichlet_distribution_from_alphas
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from collections.abc import Callable

    from flextype.isinstance import LazyType
    from probly.predictor import Predictor


@flexdispatch
def dirichlet_clipped_exp_one_activation_appender[**In, Out](base: Predictor[In, Out]) -> Predictor[In, Out]:
    """Append a clipped-exp-plus-one activation function to a base model."""
    msg = f"No Dirichlet clipped-exp-plus-one activation appender registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, appender: Callable) -> None:
    """Register a base model that the activation function will be appended to."""
    dirichlet_clipped_exp_one_activation_appender.register(cls=cls, func=appender)


@runtime_checkable
class DirichletClippedExpOneActivationPredictor[**In, Out: DirichletDistribution](EvidentialPredictor, Protocol):
    """Protocol for predictors that convert logits to Dirichlet alpha with clipped exp plus one."""


@predictor_transformation(permitted_predictor_types=[LogitClassifier], preserve_predictor_type=False)
@DirichletClippedExpOneActivationPredictor.register_factory
def dirichlet_clipped_exp_one_activation[T: Predictor](base: T) -> T:
    """Append clipped exp plus one so logits become Dirichlet concentrations.

    Args:
        base: The base logit classifier to be wrapped.

    Returns:
        A predictor returning Dirichlet alpha with values bounded below by 1.
    """
    return dirichlet_clipped_exp_one_activation_appender(base)  # ty:ignore[invalid-return-type]


@predict.register(DirichletClippedExpOneActivationPredictor)
def _[**In](
    predictor: DirichletClippedExpOneActivationPredictor[In, DirichletDistribution],
    *args: In.args,
    **kwargs: In.kwargs,
) -> DirichletDistribution:
    """Predict with a Dirichlet clipped-exp-plus-one activation predictor."""
    return create_dirichlet_distribution_from_alphas(predict_raw(predictor, *args, **kwargs))
