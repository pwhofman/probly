"""Shared implementation of Prior Networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.method import predictor_transformation
from probly.predictor import EvidentialPredictor, LogitClassifier, predict, predict_raw
from probly.representation.distribution import DirichletDistribution, create_dirichlet_distribution_from_alphas

if TYPE_CHECKING:
    from collections.abc import Callable

    from flextype.isinstance import LazyType
    from probly.predictor import Predictor


@flexdispatch
def prior_network_appender[**In, Out](base: Predictor[In, Out]) -> Predictor[In, Out]:
    """Append a Prior Network activation function to a base model."""
    msg = f"No prior network appender registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, appender: Callable) -> None:
    """Register a base model that the activation function will be appended to."""
    prior_network_appender.register(cls=cls, func=appender)


@runtime_checkable
class PriorNetworkPredictor[**In, Out: DirichletDistribution](EvidentialPredictor, Protocol):
    """Protocol for Malinin-style Prior Network predictors.

    The network outputs Dirichlet concentration parameters alpha directly,
    computed internally as alpha = exp(base(x)), following
    :cite:`malininPredictiveUncertaintyEstimation2018`.
    """


@predictor_transformation(permitted_predictor_types=[LogitClassifier], preserve_predictor_type=False)
@PriorNetworkPredictor.register_factory
def prior_network[T: Predictor](base: T) -> T:
    """Create a Prior Network predictor based on :cite:`malininPredictiveUncertaintyEstimation2018`.

    Args:
        base: The base logit classifier to be wrapped.

    Returns:
        A Prior Network predictor returning Dirichlet alpha.
    """
    return prior_network_appender(base)  # ty:ignore[invalid-return-type]


@predict.register(PriorNetworkPredictor)
def _[**In](
    predictor: PriorNetworkPredictor[In, DirichletDistribution],
    *args: In.args,
    **kwargs: In.kwargs,
) -> DirichletDistribution:
    """Predict with a Prior Network predictor."""
    return create_dirichlet_distribution_from_alphas(predict_raw(predictor, *args, **kwargs))
