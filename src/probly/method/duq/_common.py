"""Shared Deterministic Uncertainty Quantification method implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from flextype import flexdispatch
from probly.decider import categorical_from_mean
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import CachingDecomposition, TotalDecomposition
from probly.representation.distribution._common import create_categorical_distribution
from probly.representation.representation import Representation
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution import CategoricalDistribution


@runtime_checkable
class DUQRepresentation(Representation, Protocol):
    r"""Representation of a DUQ model output."""

    @property
    def kernel_values(self) -> ArrayLike:
        """Per-class RBF kernel values, shape ``(..., num_classes)``."""


@flexdispatch
def create_duq_representation(kernel_values: ArrayLike) -> DUQRepresentation:
    """Create a DUQ representation from per-class kernel values."""
    msg = f"No DUQ representation factory registered for kernel values of type {type(kernel_values)}"
    raise NotImplementedError(msg)


@runtime_checkable
class DUQPredictor[**In, Out: DUQRepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor that produces RBF-kernel centroid-head uncertainty scores."""

    encoder: Predictor[In, Out]
    centroid_head: Predictor[In, Out]


@flexdispatch
def duq_generator[**In, Out: DUQRepresentation](
    base: Predictor[In, Out],
    centroid_size: int,
    length_scale: float,
    gamma: float,
) -> DUQPredictor[In, Out]:
    """Generate an DUQ model from a base model."""
    msg = f"No DUQ generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@DUQPredictor.register_factory
def duq[**In, Out: DUQRepresentation](
    base: Predictor[In, Out],
    centroid_size: int = 256,
    length_scale: float = 0.1,
    gamma: float = 0.999,
) -> DUQPredictor[In, Out]:
    r"""Replace the final classifier head with an RBF centroid head."""
    return duq_generator(base, centroid_size, length_scale, gamma)


@flexdispatch
def duq_uncertainty(kernel_values: Iterable) -> ArrayLike:
    r"""Compute the DUQ uncertainty score :math:`1 - \max_c K_c(x)`."""
    msg = f"DUQ uncertainty is not implemented for kernel values of type {type(kernel_values)}"
    raise NotImplementedError(msg)


@decompose.register(DUQRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class DUQDecomposition[T](CachingDecomposition, TotalDecomposition[T]):
    """DUQ total-uncertainty decomposition."""

    representation: DUQRepresentation

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty of the decomposition."""
        return duq_uncertainty(self.representation.kernel_values)  # ty:ignore[invalid-return-type]


@categorical_from_mean.register(DUQRepresentation)
def _(representation: DUQRepresentation) -> CategoricalDistribution:
    return create_categorical_distribution(representation.kernel_values)


__all__ = [
    "DUQDecomposition",
    "DUQPredictor",
    "DUQRepresentation",
    "create_duq_representation",
    "duq",
    "duq_uncertainty",
]
