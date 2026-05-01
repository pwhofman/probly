"""Shared Dirichlet exp-activation transformation implementation."""

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
def dirichlet_exp_activation_appender[**In, Out](base: Predictor[In, Out]) -> Predictor[In, Out]:
    """Append an exp activation function to a base model."""
    msg = f"No Dirichlet exp-activation appender registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, appender: Callable) -> None:
    """Register a base model that the activation function will be appended to."""
    dirichlet_exp_activation_appender.register(cls=cls, func=appender)


@runtime_checkable
class DirichletExpActivationPredictor[**In, Out: DirichletDistribution](EvidentialPredictor, Protocol):
    """Protocol for predictors that convert logits to Dirichlet alpha with exp.

    The network outputs Dirichlet concentration parameters alpha directly,
    computed internally as alpha = exp(base(x)), following
    :cite:`malininPredictiveUncertaintyEstimation2018`.
    """


@predictor_transformation(permitted_predictor_types=[LogitClassifier], preserve_predictor_type=False)
@DirichletExpActivationPredictor.register_factory
def dirichlet_exp_activation[T: Predictor](base: T) -> T:
    """Append an exp activation so logits become Dirichlet concentrations.

    Args:
        base: The base logit classifier to be wrapped.

    Returns:
        A predictor returning Dirichlet alpha.
    """
    return dirichlet_exp_activation_appender(base)  # ty:ignore[invalid-return-type]


@predict.register(DirichletExpActivationPredictor)
def _[**In](
    predictor: DirichletExpActivationPredictor[In, DirichletDistribution],
    *args: In.args,
    **kwargs: In.kwargs,
) -> DirichletDistribution:
    """Predict with a Dirichlet exp-activation predictor."""
    return create_dirichlet_distribution_from_alphas(predict_raw(predictor, *args, **kwargs))
