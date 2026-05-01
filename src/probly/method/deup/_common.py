"""Shared Direct Epistemic Uncertainty Prediction (DEUP) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from flextype import flexdispatch
from probly.decider import categorical_from_mean
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.distribution import LogBase, entropy
from probly.representation.representation import Representation
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution._common import CategoricalDistribution


@runtime_checkable
class DEUPRepresentation(Representation, Protocol):
    r"""Representation of a DEUP model output.

    Holds the two quantities needed for uncertainty quantification:
    softmax probabilities (aleatoric surrogate) and a scalar predicted
    cross-entropy error score (total uncertainty estimate).

    The uncertainty decomposition is additive:

    - total = ``error_score`` -- predicted expected cross-entropy from the
      learned error predictor head.
    - aleatoric = :math:`H(\text{softmax}(x))` -- entropy of the predictive
      distribution, which approximates the Bayes (irreducible) risk.
    - epistemic = total - aleatoric -- the excess risk, i.e. the part of the
      expected error that is reducible with more data or a better model.
    """

    @property
    def softmax(self) -> CategoricalDistribution:
        """Softmax probabilities of the base classifier, shape ``(batch, num_classes)``."""

    @property
    def error_score(self) -> ArrayLike:
        """Predicted per-sample expected cross-entropy, shape ``(batch,)``."""


@flexdispatch
def create_deup_representation(softmax: CategoricalDistribution, error_score: ArrayLike) -> DEUPRepresentation:
    """Create a DEUP representation from a softmax distribution and error score.

    Args:
        softmax: Softmax probabilities of the base classifier.
        error_score: Predicted per-sample expected cross-entropy from the error head.

    Returns:
        A :class:`DEUPRepresentation` instance.
    """
    msg = (
        f"No DEUP representation factory registered for softmax type {type(softmax)}"
        f" and error_score type {type(error_score)}"
    )
    raise NotImplementedError(msg)


@runtime_checkable
class DEUPPredictor[**In, Out: DEUPRepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor combining a spectrally-normalised encoder with a learned error head.

    Components:
        encoder: Feature extractor (backbone with classification head removed).
        classification_head: The original classification linear layer.
        error_head: A small MLP trained post-hoc to predict the per-sample
            cross-entropy of the main predictor on held-out data.
    """

    encoder: Predictor[In, Out]
    classification_head: Predictor[In, Out]
    error_head: Predictor[In, Out]


@flexdispatch
def deup_generator[**In, Out: DEUPRepresentation](
    base: Predictor[In, Out],
    hidden_size: int,
    n_hidden_layers: int,
) -> DEUPPredictor[In, Out]:
    """Generate a DEUP model from a base model."""
    msg = f"No DEUP generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@DEUPPredictor.register_factory
def deup[**In, Out: DEUPRepresentation](
    base: Predictor[In, Out],
    hidden_size: int = 256,
    n_hidden_layers: int = 2,
) -> DEUPPredictor[In, Out]:
    r"""Transform a model for Direct Epistemic Uncertainty Prediction :cite:`lahlouDirectEpistemicUncertainty2023`.

    Strips the final classification head from ``base`` to obtain a feature
    encoder, then attaches two heads:

    - ``classification_head``: the original last ``nn.Linear`` layer, kept
      for phase-1 cross-entropy training.
    - ``error_head``: a small MLP of depth ``n_hidden_layers`` with
      ``hidden_size`` units and ReLU activations, mapping encoder features
      to a scalar predicted per-sample cross-entropy.

    **Training** proceeds in two phases:

    1. Train ``encoder`` and ``classification_head`` jointly with standard
       cross-entropy, identical to a plain classifier.
    2. Freeze ``encoder`` and ``classification_head``; train ``error_head``
       on held-out data (e.g. the validation set) with MSE between the
       predicted scalar and the actual per-sample cross-entropy of the
       frozen main model.

    **Uncertainty decomposition** at inference is additive:

    - total = ``error_score`` (output of ``error_head``),
    - aleatoric = :math:`H(\text{softmax}(x))` (entropy of predictive distribution),
    - epistemic = total - aleatoric (excess risk).

    Args:
        base: Base classification model to be transformed.
        hidden_size: Width of each hidden layer in the error head.
        n_hidden_layers: Number of hidden layers in the error head (minimum 1).

    Returns:
        The transformed DEUP predictor.
    """
    return deup_generator(base, hidden_size, n_hidden_layers)


@decompose.register(DEUPRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class DEUPDecomposition[T](AdditiveDecomposition[T, T, T]):
    r"""DEUP additive uncertainty decomposition.

    Decomposes total uncertainty into aleatoric and epistemic components
    following :cite:`lahlouDirectEpistemicUncertainty2023`:

    - ``total`` = predicted expected cross-entropy (output of the error head),
    - ``aleatoric`` = softmax entropy :math:`H(\text{softmax}(x))` (Bayes risk
      approximation),
    - ``epistemic`` = total - aleatoric (excess risk).

    All three quantities are exposed and the additive constraint
    total = aleatoric + epistemic holds by construction (up to the quality
    of the error head fit).
    """

    representation: DEUPRepresentation
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty: predicted expected cross-entropy."""
        return self.representation.error_score  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty: entropy of the softmax distribution."""
        return entropy(self.representation.softmax, base=self.base)

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty: excess risk (total minus aleatoric)."""
        return self.representation.error_score - entropy(self.representation.softmax, base=self.base)


@categorical_from_mean.register(DEUPRepresentation)
def _(representation: DEUPRepresentation) -> CategoricalDistribution:
    return representation.softmax


__all__ = [
    "DEUPDecomposition",
    "DEUPPredictor",
    "DEUPRepresentation",
    "create_deup_representation",
    "deup",
    "deup_generator",
]
