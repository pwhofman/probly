"""Shared Mahalanobis out-of-distribution detection method implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from flextype import flexdispatch
from probly.decider import categorical_from_mean
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import CachingDecomposition, EpistemicDecomposition
from probly.representation.representation import Representation
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution._common import CategoricalDistribution

type FeatureNodes = Sequence[str] | None


@runtime_checkable
class MahalanobisRepresentation(Representation, Protocol):
    """Representation of a Mahalanobis OOD model output.

    Holds the softmax probabilities (used for the class prediction) and the
    per-layer Mahalanobis confidence scores together with the logistic-regression
    combination weights used to turn them into a single epistemic/OOD score.
    """

    @property
    def softmax(self) -> CategoricalDistribution:
        """Softmax probabilities."""

    @property
    def layer_scores(self) -> ArrayLike:
        """Per-layer Mahalanobis confidence, shape ``(..., num_layers)``."""

    @property
    def weight(self) -> ArrayLike:
        """Logistic-regression combination weights, shape ``(num_layers,)``."""

    @property
    def bias(self) -> ArrayLike:
        """Logistic-regression combination bias (scalar)."""


@flexdispatch
def create_mahalanobis_representation(
    softmax: CategoricalDistribution,
    layer_scores: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike,
) -> MahalanobisRepresentation:
    """Create a Mahalanobis representation from softmax, layer scores and combiner weights."""
    msg = f"No Mahalanobis representation factory registered for softmax type {type(softmax)}"
    raise NotImplementedError(msg)


@runtime_checkable
class MahalanobisPredictor[**In, Out: MahalanobisRepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor that scores inputs by class-conditional Mahalanobis distance."""

    encoder: Predictor[In, Out]
    classification_head: Predictor[In, Out]


@flexdispatch
def mahalanobis_generator[**In, Out: MahalanobisRepresentation](
    base: Predictor[In, Out],
    feature_nodes: FeatureNodes,
    input_preprocessing_eps: float,
) -> MahalanobisPredictor[In, Out]:
    """Generate a Mahalanobis model from a base model."""
    msg = f"No Mahalanobis generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@MahalanobisPredictor.register_factory
def mahalanobis[**In, Out: MahalanobisRepresentation](
    base: Predictor[In, Out],
    feature_nodes: FeatureNodes = None,
    input_preprocessing_eps: float = 0.0,
) -> MahalanobisPredictor[In, Out]:
    """Turn a classifier into a Mahalanobis-distance OOD detector.

    Based on :cite:`leeSimpleUnifiedFramework2018`.  The final linear head is
    stripped to expose penultimate features, class-conditional Gaussians with a
    tied covariance are fitted on those features (and, optionally, on extra
    intermediate layers), and the per-layer Mahalanobis confidence is combined
    into a single out-of-distribution score by logistic regression.

    The returned predictor still needs its Gaussian parameters fitted after
    training via ``fit_mahalanobis_heads`` (and, optionally, ``fit_combiner``
    to calibrate the multi-layer weights on in- vs out-of-distribution data).

    Args:
        base: Base logit classifier to be transformed.
        feature_nodes: Optional names of intermediate submodules (as returned by
            ``named_modules``) whose global-average-pooled outputs provide
            additional feature layers for the ensemble. When ``None`` only the
            penultimate features are used, recovering the single-layer detector.
        input_preprocessing_eps: Magnitude of the FGSM-style input perturbation
            applied at inference to sharpen the in/out separation. ``0`` disables
            preprocessing.

    Returns:
        The transformed Mahalanobis predictor.
    """
    return mahalanobis_generator(base, feature_nodes, input_preprocessing_eps)


@flexdispatch
def combine_layer_scores(layer_scores: ArrayLike, weight: ArrayLike, bias: ArrayLike) -> ArrayLike:
    """Combine per-layer Mahalanobis confidences into a single OOD score."""
    msg = f"Combining layer scores is not supported for scores of type {type(layer_scores)}."
    raise NotImplementedError(msg)


@decompose.register(MahalanobisRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class MahalanobisDecomposition[T](CachingDecomposition, EpistemicDecomposition[T]):
    """Mahalanobis decomposition exposing the combined OOD score as epistemic uncertainty.

    Following :cite:`leeSimpleUnifiedFramework2018`, which proposes only the
    Mahalanobis OOD score (no aleatoric or total measure), this decomposition has
    a single epistemic slot, and its canonical notion is epistemic uncertainty.
    """

    representation: MahalanobisRepresentation

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty: the combined per-layer Mahalanobis OOD score."""
        return combine_layer_scores(  # ty:ignore[invalid-return-type]
            self.representation.layer_scores,
            self.representation.weight,
            self.representation.bias,
        )


@categorical_from_mean.register(MahalanobisRepresentation)
def _(representation: MahalanobisRepresentation) -> CategoricalDistribution:
    return representation.softmax


__all__ = [
    "MahalanobisDecomposition",
    "MahalanobisPredictor",
    "MahalanobisRepresentation",
    "combine_layer_scores",
    "create_mahalanobis_representation",
    "mahalanobis",
    "mahalanobis_generator",
]
