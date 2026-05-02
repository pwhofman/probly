"""Shared Direct Epistemic Uncertainty Prediction (DEUP) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, override, runtime_checkable

from flextype import flexdispatch
from probly.decider import categorical_from_mean
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.representation.representation import Representation
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.quantification.measure.distribution import LogBase
    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution._common import CategoricalDistribution


@runtime_checkable
class DEUPRepresentation(Representation, Protocol):
    r"""Representation of a DEUP model output.

    Holds the two quantities needed for uncertainty quantification:
    softmax probabilities of the base classifier and a scalar predicted
    cross-entropy error score from the DEUP error head.

    Following the paper's classification setup
    (:cite:`lahlou2021deup`, Sec. 4.3.1), the
    aleatoric component is taken to be zero -- estimating
    :math:`H[P(Y|x)]` of the *ground-truth* conditional would require
    label replicates which are unavailable for standard image
    classification.  Total and epistemic uncertainty therefore both
    coincide with the predicted error score.
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
    """A predictor combining a feature encoder with a learned error head.

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
    stationarizing_features: Sequence[str | dict[str, Any]] | None = None,
    sn_coeff: float | None = None,
) -> DEUPPredictor[In, Out]:
    """Generate a DEUP model from a base model."""
    msg = f"No DEUP generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@DEUPPredictor.register_factory
def deup[**In, Out: DEUPRepresentation](
    base: Predictor[In, Out],
    hidden_size: int = 1024,
    n_hidden_layers: int = 5,
    stationarizing_features: Sequence[str | dict[str, Any]] | None = None,
    sn_coeff: float | None = None,
) -> DEUPPredictor[In, Out]:
    r"""Transform a model for Direct Epistemic Uncertainty Prediction :cite:`lahlou2021deup`.

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

    **Uncertainty decomposition** at inference (classification setup
    of :cite:`lahlou2021deup`):

    - total = ``error_score`` (output of ``error_head``),
    - aleatoric = 0 (ground-truth label entropy, unavailable without replicates),
    - epistemic = ``error_score`` (the reducible part of the loss).

    .. note::
       The default stationarizing features (``log_maf_density`` +
       ``log_due_variance``) match the paper's CIFAR-10 setup in spirit:
       a normalizing-flow density estimate :math:`\log \hat{q}(z|D)` in
       encoder feature space, and a DUE posterior variance
       :math:`\log \hat{V}(x)` from a spectrally-normalized encoder +
       SVGP.  The paper uses a MAF on raw pixels and DUE with full GP;
       here both operate on encoder features for scalability.

    Args:
        base: Base classification model to be transformed.
        hidden_size: Width of each hidden layer in the error head.
        n_hidden_layers: Number of hidden layers in the error head (minimum 1).
        stationarizing_features: Sequence of stationarizing feature
            specifications (see :cite:`lahlou2021deup`).  Each entry is
            either a registry name (e.g. ``"log_maf_density"``,
            ``"log_due_variance"``) or a mapping ``{"name": ..., **kwargs}``.
            The error head receives **only** these features — encoder features
            are excluded.  At least one provider is required.
        sn_coeff: Lipschitz coefficient for spectral normalization of the
            encoder.  Defaults to ``None`` (no spectral norm).  Automatically
            set to 3.0 when a provider that requires spectral norm is used
            (e.g. ``log_due_variance``), matching the DUE / DDU convention.

    Returns:
        The transformed DEUP predictor.
    """
    return deup_generator(base, hidden_size, n_hidden_layers, stationarizing_features, sn_coeff)


@decompose.register(DEUPRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class DEUPDecomposition[T](AdditiveDecomposition[T, T, T]):
    r"""DEUP additive uncertainty decomposition.

    Follows the classification setup of
    :cite:`lahlou2021deup`, where
    aleatoric uncertainty :math:`H[P(Y|x)]` of the ground-truth
    conditional is taken to be zero in the absence of label replicates:

    - ``total`` = predicted expected cross-entropy (output of the error head),
    - ``aleatoric`` = 0,
    - ``epistemic`` = total = predicted expected cross-entropy.

    The additive constraint total = aleatoric + epistemic therefore holds
    by construction.  Note that, contrary to a common shortcut, the
    entropy of the predictive softmax is *not* the aleatoric
    uncertainty in the DEUP definition: the paper defines aleatoric as
    the entropy of the (unknown) ground-truth conditional, not of the
    learner's estimate.
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
        """The aleatoric uncertainty: zero in the standard classification setup."""
        return cast("Any", self.representation.error_score) * 0


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
