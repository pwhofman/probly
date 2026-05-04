"""Shared SNGP implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable
import warnings

from flextype import flexdispatch
from probly.predictor import Predictor, RandomPredictor, predict, predict_raw
from probly.quantification.decomposition.decomposition import CachingDecomposition, EpistemicDecomposition
from probly.quantification.measure.distribution import DEFAULT_MEAN_FIELD_FACTOR, dempster_shafer_uncertainty
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
    from probly.representation.sample import Sample

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

NUM_RANDOM_FEATURES = GlobalVariable[int](
    "NUM_RANDOM_FEATURES",
    "Dimensionality of the random Fourier feature map (D_L in the SNGP paper, Eq. 7). "
    "Independent of the replaced Linear's in_features. Default is 1024 (imagenet recipe).",
)
RANDOM_FEATURE_INIT_STD = GlobalVariable[float](
    "RANDOM_FEATURE_INIT_STD", "Standard deviation of the Gaussian used to init the frozen random projection W_L. "
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


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@SNGPPredictor.register_factory
def sngp[**In, Out: GaussianDistribution](
    base: Predictor[In, Out],
    name: str = "weight",
    n_power_iterations: int = 1,
    norm_multiplier: float = 6.0,
    eps: float = 1e-12,
    num_random_features: int = 1024,
    ridge_penalty: float = 1.0,
    momentum: float = -1.0,
    random_feature_init_std: float = 1.0,
) -> SNGPPredictor[In, Out]:
    """Wrap a model with SNGP (Spectral-normalized Neural Gaussian Process).

    Replaces the last ``nn.Linear`` with an :class:`SNGPLayer` (random
    Fourier features + Laplace-approximated Gaussian process) and registers
    a spectral-norm parametrization on every preceding ``nn.Linear`` and
    ``nn.Conv2d``. Defaults match the ImageNet ResNet-50
    baseline at ``google/uncertainty-baselines/baselines/imagenet/sngp.py``.

    Args:
        base: The model to wrap.
        name: The name of the weight parameter to spectrally normalize on
            non-output layers. Defaults to ``"weight"``.
        n_power_iterations: Power iterations per training step for the
            spectral-norm estimate. Defaults to 1. (A 15-iteration warmup
            runs once at construction time, independent of this value.)
        norm_multiplier: Upper bound on each non-output layer's spectral
            norm. Defaults to 6.0 (high because ResNet's BatchNorm already
            applies its own Lipschitz scaling).
        eps: Small constant to stabilize the spectral-norm denominator.
            Defaults to 1e-12.
        num_random_features: Dimensionality of the random Fourier feature
            map. Defaults to ``1024``.
        ridge_penalty: Ridge factor used inside the covariance inversion
            ``inv(ridge * I + precision)``. Defaults to 1.0.
        momentum: Discount factor for the precision-matrix update. Default
            ``-1.0`` triggers accumulate mode (paper Algorithm 1; imagenet
            ``gp_cov_discount_factor=-1``); the user **must** call
            :func:`reset_precision_matrix` at the start of each training
            epoch in this mode. Pass ``momentum > 0`` for EMA mode (no
            reset needed; matches the CLINC reference's
            ``gp_cov_momentum=0.999``).
        random_feature_init_std: Standard deviation of the Gaussian used to
            initialize the frozen random projection ``W_L``. Defaults to
            ``1.0`` (paper / imagenet / Edward2; full RFF kernel
            approximation, expects from-scratch training). Set to ``0.05``
            (matching ``untangle.wrappers.sngp_wrapper``) when fine-tuning
            from a pretrained backbone: keeps ``W_L^T h`` in the near-linear
            regime of ``cos`` so pretrained-feature signal flows through
            the RFF, at the cost of a longer effective kernel lengthscale
            and weaker distance-aware uncertainty.

    Returns:
        An ``SNGPPredictor`` whose ``predict(...)`` returns a
        ``GaussianDistribution`` over logits.
    """
    transformed = traverse(
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
            NUM_RANDOM_FEATURES: num_random_features,
            RIDGE_PENALTY: ridge_penalty,
            MOMENTUM: momentum,
            RANDOM_FEATURE_INIT_STD: random_feature_init_std,
        },
    )
    skipped = _collect_skipped_param_bearing_layer_classes(transformed)
    if skipped:
        warnings.warn(
            f"sngp(): the following parameter-bearing layer types were not "
            f"spectrally normalized and will violate the bi-Lipschitz "
            f"property of the hidden mapping (paper Eq. 5): {skipped}. "
            f"SNGP currently only wraps nn.Linear and nn.Conv2d.",
            stacklevel=2,
        )
    return transformed


@predict.register(SNGPPredictor)
def _[**In](
    predictor: SNGPPredictor[In, GaussianDistribution], *args: In.args, **kwargs: In.kwargs
) -> GaussianDistribution:
    """Predict method for SNGP predictors."""
    logits, variance = predict_raw(predictor, *args, **kwargs)
    distribution = create_gaussian_distribution(logits, variance)
    return distribution


representer.register(SNGPPredictor, SNGPRepresenter)


@flexdispatch
def _collect_skipped_param_bearing_layer_classes(predictor: Any) -> list[str]:  # noqa: ANN401
    """Return sorted unique names of param-bearing layer classes the SNGP traverser skipped.

    Backend-agnostic dispatch. Each backend (torch, future flax) registers its
    own walker that knows how to enumerate submodules and detect which ones
    are "handled" (either wrapped by the SNGP traverser or intentionally
    excluded, like norm layers).

    Anything with direct parameters that is **not** in the handled set is
    "skipped" - i.e., its weights are not spectrally normalized - and is
    reported so the caller knows the bi-Lipschitz property of the hidden
    mapping (paper Eq. 5) is not enforced for those layers.
    """
    msg = f"_collect_skipped_param_bearing_layer_classes not implemented for type {type(predictor).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def reset_precision_matrix(predictor: Any) -> None:  # noqa: ANN401
    """Zero the precision matrix of every SNGP layer in ``predictor``.

    Call at the start of each training epoch when using the default
    ``momentum=-1`` (accumulate) mode. With ``momentum > 0`` (EMA mode) this
    is rarely needed; the EMA naturally bounds the precision matrix.

    Backend-agnostic dispatch. The torch implementation walks
    ``predictor.modules()`` and resets every
    :class:`probly.layers.torch.SNGPLayer` it finds. If no SNGP layers are
    present the implementation emits a DEBUG log message and returns
    silently (common cause: caller passed an un-wrapped model rather than
    the result of :func:`sngp`).

    Args:
        predictor: A predictor returned by :func:`sngp`.
    """
    msg = f"reset_precision_matrix not implemented for type {type(predictor).__name__}."
    raise NotImplementedError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SNGPDecomposition[T](CachingDecomposition, EpistemicDecomposition[T]):
    """Decomposition based on SNGP's Dempster-Shafer epistemic uncertainty.

    Implements the single uncertainty quantity used by SNGP
    :cite:`liuSNGPSpectralNormalizedNeural2020` for OOD detection (Sec. 5.2 +
    Appendix C, Eq. 15). For an SNGP-style Gaussian distribution
    ``N(h, sigma^2)`` over K-class logits:

    - Epistemic uncertainty: ``K / (K + sum_k exp(h_adj_k))`` where
      ``h_adj_k = h_k / sqrt(1 + mean_field_factor * sigma_k^2)``. This is the
      Dempster-Shafer / vacuity metric (originally introduced by
      :cite:`sensoyEvidentialDeepLearning2018`), applied to a soft-evidential
      Dirichlet with ``alpha = 1 + exp(h_adj)``. The mean-field correction
      shrinks the logits toward zero when the GP variance is large (OOD),
      driving the score toward 1.

    The paper does not propose an aleatoric or total uncertainty measure; the
    DS metric is the only OOD score used. Consequently this decomposition has
    only an epistemic slot, and its canonical notion is epistemic uncertainty.

    Args:
        distribution: A Gaussian distribution over K-class logits (e.g. the
            output of :func:`predict` on an :class:`SNGPPredictor`).
        mean_field_factor: Coefficient in front of the variance in the
            mean-field denominator. Defaults to ``pi / 8`` (paper / SNGP recipe).
            Set to ``0.0`` to ignore variance and compute the literal Eq. 15
            on the raw GP mean.
    """

    distribution: GaussianDistribution
    mean_field_factor: float = DEFAULT_MEAN_FIELD_FACTOR

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty: SNGP's mean-field-adjusted Dempster-Shafer score."""
        return dempster_shafer_uncertainty(self.distribution, mean_field_factor=self.mean_field_factor)  # ty:ignore[invalid-return-type]
