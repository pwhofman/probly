"""Shared SNGP implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.predictor import Predictor, RandomPredictor, predict, predict_raw
from probly.representation.distribution import GaussianDistribution, create_gaussian_distribution
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

sngp_traverser = flexdispatch_traverser[object](name="sngp_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER", "Whether the current layer is the last layer of the model.")

NAME = GlobalVariable[str]("NAME", "The name of the weight parameter")
N_POWER_ITERATIONS = GlobalVariable[int](
    "N_POWER_ITERATIONS", "The number of power iterations to perform for spectral normalization."
)
NORM_MULTIPLIER = GlobalVariable[float]("NORM_MULTIPLIER", "The multiplier for the spectral norm. Default is 1.0.")
EPS = GlobalVariable[float](
    "EPS", "A small value to prevent division by zero in spectral normalization. Default is 1e-12."
)

NUM_INDUCING = GlobalVariable[int](
    "NUM_INDUCING", "The number of inducing points to use for the Gaussian process layer. Default is 128."
)
RIDGE_PENALTY = GlobalVariable[float](
    "RIDGE_PENALTY",
    "The ridge penalty to apply to the covariance matrix in the Gaussian process layer. Default is 1e-6.",
)
MOMENTUM = GlobalVariable[float](
    "MOMENTUM",
    "The momentum to use for updating the covariance matrix in the Gaussian process layer. Default is 0.999.",
)


@runtime_checkable
class SNGPPredictor[**In, Out: GaussianDistribution](RandomPredictor[In, Out], Protocol):
    """A predictor that applies the SNGP representer."""


def register(cls: type, traverser: flexdispatch_traverser) -> None:
    """Register a class to be transformed by SNGP."""
    traverser.register(cls=cls, traverser=sngp_traverser, vars={CLONE: True})


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)  # ty: ignore[invalid-argument-type]
@SNGPPredictor.register_factory
def sngp[**In, Out: GaussianDistribution](
    base: Predictor[In, Out],
    name: str = "weight",
    n_power_iterations: int = 1,
    norm_multiplier: float = 1.0,
    eps: float = 1e-12,
    num_inducing: int = 128,
    ridge_penalty: float = 1e-6,
    momentum: float = 0.999,
) -> SNGPPredictor[In, Out]:
    return traverse(
        base,
        nn_compose(sngp_traverser),
        init={
            CLONE: True,
            TRAVERSE_REVERSED: True,
            LAST_LAYER: True,
            NAME: name,
            N_POWER_ITERATIONS: n_power_iterations,
            NORM_MULTIPLIER: norm_multiplier,
            EPS: eps,
            NUM_INDUCING: num_inducing,
            RIDGE_PENALTY: ridge_penalty,
            MOMENTUM: momentum,
        },
    )


@predict.register(SNGPPredictor)
def _[**In](
    predictor: SNGPPredictor[In, GaussianDistribution], *args: In.args, **kwargs: In.kwargs
) -> GaussianDistribution:
    """Predict method for SNGP predictors."""
    logits, variance = predict_raw(predictor, *args, **kwargs)
    distribution = create_gaussian_distribution(logits, variance)
    return distribution
