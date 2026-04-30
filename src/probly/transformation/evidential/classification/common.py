"""Shared evidential classification implementation."""

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
def evidential_classification_appender[**In, Out](base: Predictor[In, Out]) -> Predictor[In, Out]:
    """Append an evidential classification activation function to a base model."""
    msg = f"No evidential classification appender registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, appender: Callable) -> None:
    """Register a base model that the activation function will be appended to."""
    evidential_classification_appender.register(cls=cls, func=appender)


@runtime_checkable
class EvidentialClassificationPredictor[**In, Out: DirichletDistribution](EvidentialPredictor, Protocol):
    """Protocol for Sensoy-like evidential classification predictors.

    The network outputs Dirichlet concentration parameters alpha directly,
    computed internally as alpha = softplus(base(x)) + 1.
    """


@predictor_transformation(permitted_predictor_types=[LogitClassifier], preserve_predictor_type=False)
@EvidentialClassificationPredictor.register_factory
def evidential_classification[T: Predictor](base: T) -> T:
    """Create an evidential classification predictor based on :cite:`sensoyEvidentialDeep2018`.

    Args:
        base: The base logit classifier to be wrapped.

    Returns:
        An evidential classification predictor returning Dirichlet alpha.
    """
    return evidential_classification_appender(base)  # ty:ignore[invalid-return-type]


@predict.register(EvidentialClassificationPredictor)
def _[**In](
    predictor: EvidentialClassificationPredictor[In, DirichletDistribution],
    *args: In.args,
    **kwargs: In.kwargs,
) -> DirichletDistribution:
    """Predict with an evidential classification predictor."""
    return create_dirichlet_distribution_from_alphas(predict_raw(predictor, *args, **kwargs))
