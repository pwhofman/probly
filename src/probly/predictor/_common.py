"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from flextype import ProtocolRegistry, flexdispatch
from probly.representation import Representation
from probly.representation.credal_set import CredalSet, ProbabilityIntervalsCredalSet
from probly.representation.distribution import (
    CategoricalDistribution,
    DirichletDistribution,
    Distribution,
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
)
from probly.utils.switchdispatch import switch

type PredictorName = Literal[
    "categorical_distribution_predictor",
    "probabilistic_classifier",
    "logit_distribution_predictor",
    "logit_classifier",
    "dirichlet_distribution_predictor",
    "evidential_classifier",
]


# Protocols for predictors


@runtime_checkable
class Predictor[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for generic predictors."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        predict_method = getattr(subclass, "predict", None)
        if predict_method is not None and callable(predict_method):
            return True
        if issubclass(subclass, Callable):
            return True
        return NotImplemented


predictor_registry = switch[PredictorName, type[Predictor]]()


@runtime_checkable
class RandomPredictor[**In, Out](Predictor[In, Out], Protocol):
    """Protocol for non-deterministic predictors."""


@runtime_checkable
class IterablePredictor[**In, Out](Predictor[In, Iterable[Out]], Protocol):
    """Protocol for predictors that return an iterable of outputs."""


@runtime_checkable
class RepresentationPredictor[**In, Out: Representation](Predictor[In, Out], Protocol):
    """Protocol for predictors that return a distribution over outputs."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        predict_method = getattr(subclass, "predict_representation", None)
        if predict_method is not None and callable(predict_method):
            return True
        return NotImplemented


@runtime_checkable  # ty:ignore[conflicting-metaclass]
class RandomRepresentationPredictor[**In, Out: Representation](
    RepresentationPredictor[In, Out], RandomPredictor[In, Out], Protocol
):
    """Protocol for non-deterministic predictors that return a distribution over outputs."""

    _running_instancehook: ClassVar[ContextVar[object]] = ContextVar(
        "RandomRepresentationPredictor._running_instancehook", default=NotImplemented
    )
    sample_space: ClassVar[type[Distribution]] = Distribution

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        if cls._running_instancehook.get() is instance:
            return NotImplemented
        try:
            tok = cls._running_instancehook.set(instance)
            if isinstance(instance, RepresentationPredictor) and isinstance(instance, RandomPredictor):
                return True
        finally:
            cls._running_instancehook.reset(tok)
        return NotImplemented


@runtime_checkable
class DistributionPredictor[**In, Out: Distribution](RepresentationPredictor[In, Out], Protocol):
    """Protocol for predictors that return a distribution over outputs."""


@predictor_registry.multi_register(["categorical_distribution_predictor", "probabilistic_classifier"])
@runtime_checkable
class CategoricalDistributionPredictor[**In, Out: CategoricalDistribution](DistributionPredictor[In, Out], Protocol):
    """Protocol for predictors that return a categorical distribution over outputs expressed as probabilities."""

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        predict_proba_method = getattr(instance, "predict_proba", None)
        if (
            predict_proba_method is not None
            and callable(predict_proba_method)
            and not hasattr(instance, "predict_representation")
        ):
            return True
        return NotImplemented


@predictor_registry.multi_register(["logit_distribution_predictor", "logit_classifier"])
@runtime_checkable
class LogitDistributionPredictor[**In, Out: CategoricalDistribution](DistributionPredictor[In, Out], Protocol):
    """Protocol for predictors that return a categorical distribution over outputs expressed as logits."""


@predictor_registry.multi_register(["dirichlet_distribution_predictor", "evidential_classifier"])
@runtime_checkable
class DirichletDistributionPredictor[**In, Out: DirichletDistribution](DistributionPredictor[In, Out], Protocol):
    """Protocol for predictors that return a Dirichlet distribution over outputs."""


@runtime_checkable
class CredalPredictor[**In, Out: CredalSet](RepresentationPredictor[In, Out], Protocol):
    """Protocol for predictors that return a set of distributions over outputs."""


@runtime_checkable
class ProbabilityIntervalPredictor[**In, Out: ProbabilityIntervalsCredalSet](CredalPredictor[In, Out], Protocol):
    """Protocol for predictors that return a set of distributions over outputs."""


# Prediction functions


@flexdispatch
def predict_raw[**In, Out](predictor: Predictor[In, Out], /, *args: In.args, **kwargs: In.kwargs) -> Any:  # noqa: ANN401
    """Calls a predictor and returns the result as-is.

    This function should only be used when the caller needs to access the raw output of the predictor,
    without any conversion to a specific type. For most use cases, the `predict` function should be used instead,
    which will attempt to convert the output to the correct type using registered conversion functions.
    """
    if isinstance(predictor, RepresentationPredictor) and hasattr(predictor, "predict_representation"):
        return predictor.predict_representation(*args, **kwargs)  # ty:ignore[call-non-callable]
    if isinstance(predictor, CategoricalDistributionPredictor) and hasattr(predictor, "predict_proba"):
        return predictor.predict_proba(*args, **kwargs)  # ty:ignore[call-non-callable]
    if hasattr(predictor, "predict"):
        return predictor.predict(*args, **kwargs)  # ty: ignore[call-non-callable]
    if callable(predictor):
        return predictor(*args, **kwargs)  # ty:ignore[call-top-callable]
    msg = f"No predict function registered for type {type(predictor)}"
    raise NotImplementedError(msg)


@flexdispatch
def predict[**In, Out](predictor: Predictor[In, Out], /, *args: In.args, **kwargs: In.kwargs) -> Out:
    """Calls a predictor via `predict_raw` and returns the result as specified in the predictor's signature.

    If the predictor does not directly return a prediction of the correct type,
    this function will attempt to convert the output to the correct type using registered conversion functions.
    """
    return predict_raw(predictor, *args, **kwargs)


@predict.register(RepresentationPredictor)
def predict_representation[**In, Out](
    predictor: RepresentationPredictor[In, Out], *args: In.args, **kwargs: In.kwargs
) -> Out:
    """Predict for a representation predictor."""
    raw_prediction = predict_raw(predictor, *args, **kwargs)
    if isinstance(raw_prediction, Representation):
        return raw_prediction
    msg = f"Expected predictor of type {type(predictor)} to return a Representation, but got {type(raw_prediction)}"
    raise TypeError(msg)


@predict.register(CategoricalDistributionPredictor)
def predict_categorical_distribution[**In, Out: CategoricalDistribution](
    predictor: CategoricalDistributionPredictor[In, Out], *args: In.args, **kwargs: In.kwargs
) -> Out:
    """Predict for a categorical distribution predictor."""
    return create_categorical_distribution(predict_raw(predictor, *args, **kwargs))  # ty:ignore[invalid-return-type]


@predict.register(LogitDistributionPredictor)
def predict_categorical_distribution_from_logit[**In, Out: CategoricalDistribution](
    predictor: CategoricalDistributionPredictor[In, Out], *args: In.args, **kwargs: In.kwargs
) -> Out:
    """Predict for a categorical distribution predictor."""
    return create_categorical_distribution_from_logits(predict_raw(predictor, *args, **kwargs))  # ty:ignore[invalid-return-type]
