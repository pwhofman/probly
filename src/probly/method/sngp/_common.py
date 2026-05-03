"""Shared SNGP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

from flextype import flexdispatch
from probly.predictor import Predictor, RandomPredictor, predict, predict_raw
from probly.representation.distribution import (
    CategoricalDistributionSample,
    GaussianDistribution,
    create_gaussian_distribution,
)
from probly.representer import Representer, representer
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs, RngStream

    from probly.representation.sample import Sample

type RNG = int | Rngs | RngStream

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

RNGS = GlobalVariable[RNG]("RNGS", "rngs for flax SNGP layer initialization.", default=1)


@runtime_checkable
class SNGPPredictor[**In, Out: GaussianDistribution](RandomPredictor[In, Out], Protocol):
    """A predictor that applies the SNGP representer."""


@flexdispatch
def compute_categorical_sample_from_logits(sample: Sample[Any]) -> CategoricalDistributionSample[Any]:
    """Convert a sample of SNGP logits to a categorical distribution sample."""
    msg = f"compute_categorical_sample_from_logits not implemented for type {type(sample)}."
    raise NotImplementedError(msg)


class SNGPRepresenter[**In, Out](Representer[Any, In, Out, CategoricalDistributionSample[Any]]):
    """Representer that samples SNGP logits and converts them to categorical samples."""

    num_samples: int

    def __init__(
        self,
        predictor: Predictor[In, Out],
        num_samples: int = 10,
        *args: In.args,
        **kwargs: In.kwargs,
    ) -> None:
        """Initialize the SNGP representer."""
        super().__init__(predictor, *args, **kwargs)
        self.num_samples = num_samples

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Predict the outputs from the SNGP predictor."""
        return predict(self.predictor, *args, **kwargs)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> CategoricalDistributionSample[Any]:
        distribution = self._predict(*args, **kwargs)
        sampled_logits = distribution.sample(self.num_samples)  # ty:ignore[unresolved-attribute]

        return compute_categorical_sample_from_logits(sampled_logits)


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
    rngs: RNG = 1,
) -> SNGPPredictor[In, Out]:
    """Create an SNGP predictor from a base predictor based on :cite:`liu2020SNGP`.

    Args:
        base: The base model to be transformed.
        name: The name of the kernel/weight parameter on wrapped layers used by the
            spectral-norm wrapper. Defaults to ``"weight"`` for torch and is mapped
            to ``"kernel"`` by the flax backend.
        n_power_iterations: Number of power-iteration steps for spectral norm.
        norm_multiplier: Upper bound for the spectral-norm multiplier.
        eps: Numerical-stability floor for division by ``sigma`` in spectral norm.
        num_inducing: Number of inducing features for the Gaussian-process head.
        ridge_penalty: Ridge penalty added to the precision matrix at initialization.
        momentum: EMA momentum for the precision-matrix update.
        rngs: Optional rngs for the flax SNGP layer initialization (types:
            ``rnglib.Rngs | rnglib.RngStream | int``). Ignored by the torch backend.
            Default is ``1``.

    Returns:
        The SNGP predictor.
    """
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
            RNGS: rngs,
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


representer.register(SNGPPredictor, SNGPRepresenter)
