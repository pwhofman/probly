"""Shared Deep Deterministic Uncertainty method implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from flextype import flexdispatch
from probly.decider import categorical_from_mean
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import AleatoricEpistemicDecomposition, CachingDecomposition
from probly.quantification.measure.distribution import LogBase, entropy
from probly.representation.representation import Representation
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution._common import CategoricalDistribution


@runtime_checkable
class DDURepresentation(Representation, Protocol):
    """Representation of a DDU model output.

    Holds the two quantities needed for uncertainty quantification: softmax
    probabilities (aleatoric) and density vectors (epistemic).
    """

    @property
    def softmax(self) -> CategoricalDistribution:
        """Softmax probabilities."""

    @property
    def densities(self) -> Iterable:
        """Density vectors."""


@flexdispatch
def create_ddu_representation(softmax: CategoricalDistribution, densities: Iterable) -> DDURepresentation:
    """Create a DDU representation from a softmax distribution and density vector."""
    msg = (
        f"No DDU representation factory registered for softmax type {type(softmax)} and density type {type(densities)}"
    )
    raise NotImplementedError(msg)


@runtime_checkable
class DDUPredictor[**In, Out: DDURepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor with a spectral-normalized encoder and Gaussian-mixture density head."""

    encoder: Predictor[In, Out]
    classification_head: Predictor[In, Out]
    density_head: Predictor[In, Out]


@flexdispatch
def ddu_generator[**In, Out: DDURepresentation](
    base: Predictor[In, Out],
    sn_coeff: float,
) -> DDUPredictor[In, Out]:
    """Generate a DDU model from a base model."""
    msg = f"No DDU generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@DDUPredictor.register_factory
def ddu[**In, Out: DDURepresentation](
    base: Predictor[In, Out],
    sn_coeff: float = 3.0,
) -> DDUPredictor[In, Out]:
    """Apply spectral normalization and add a Gaussian-mixture density head.

    Args:
        base: Base classification model to be transformed.
        sn_coeff: Lipschitz coefficient for spectral normalization.

    Returns:
        The transformed DDU predictor.
    """
    return ddu_generator(base, sn_coeff)


@flexdispatch
def negative_log_density(densities: Iterable) -> ArrayLike:
    """Convert DDU log-density scores to an epistemic uncertainty score."""
    msg = f"Negative log density is not supported for densities of type {type(densities)}."
    raise NotImplementedError(msg)


@decompose.register(DDURepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class DDUDensityDecomposition[T](CachingDecomposition, AleatoricEpistemicDecomposition[T, T]):
    """DDU decomposition into softmax entropy and negative feature log density."""

    representation: DDURepresentation
    base: LogBase = None

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return entropy(self.representation.softmax, base=self.base)

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty of the decomposition."""
        return negative_log_density(self.representation.densities)  # ty:ignore[invalid-return-type]


@categorical_from_mean.register(DDURepresentation)
def _(representation: DDURepresentation) -> CategoricalDistribution:
    return representation.softmax
