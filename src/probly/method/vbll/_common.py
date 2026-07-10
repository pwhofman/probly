"""Shared Variational Bayesian Last Layer (VBLL) implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

from flextype import flexdispatch

from probly.predictor import (
    Predictor,
    RandomPredictor,
    predict,
    predict_raw,
)
from probly.representation.distribution import (
    CategoricalDistributionSample,
    GaussianDistribution,
    create_gaussian_distribution,
)
from probly.representer import Representer, representer
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import find_layer, nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from probly.representation.sample import Sample

vbll_traverser = flexdispatch_traverser[object](name="vbll_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER", "Whether the current layer is the last layer of the model.")
PARAMETERIZATION = GlobalVariable[str](
    "PARAMETERIZATION", "The posterior covariance parametrization ('diagonal', 'dense' or 'lowrank')."
)
PRIOR_SCALE = GlobalVariable[float]("PRIOR_SCALE", "The scale of the isotropic prior covariance.")
NOISE_INIT = GlobalVariable[float]("NOISE_INIT", "The initial per-output noise standard deviation.")
COV_RANK = GlobalVariable[int]("COV_RANK", "The rank of the low-rank covariance factor.")
WISHART_SCALE = GlobalVariable[float]("WISHART_SCALE", "The scale of the Wishart prior on the noise precision.")
DOF = GlobalVariable[float]("DOF", "The degrees of freedom of the Wishart prior on the noise precision.")
VARIANT = GlobalVariable[str]("VARIANT", "The discriminative VBLL last-layer variant.")
NOISE_PRIOR_SCALE = GlobalVariable[float]("NOISE_PRIOR_SCALE", "The scale of the prior on the heteroscedastic noise.")


@runtime_checkable
class VBLLPredictor[**In, Out: GaussianDistribution](RandomPredictor[In, Out], Protocol):
    """A predictor with a variational Bayesian last layer.

    Its :func:`predict` returns the closed-form :class:`GaussianDistribution`
    over the network outputs (the regression predictive) or logits (for
    classification). For classification, sample-based class probabilities are
    obtained through the registered :class:`VBLLRepresenter`.
    """


def find_vbll_layer(model: object) -> Any:  # noqa: ANN401, the concrete layer type depends on the variant
    """Return the variational Bayesian last layer of a transformed predictor.

    Convenience wrapper around :func:`probly.traverse_nn.find_layer` that matches
    any discriminative VBLL layer variant (standard, Student-t, or
    heteroscedastic), e.g. to pass the layer to :func:`probly.train.vbll.vbll_loss`
    or to attach hooks to it. For generative VBLL models use
    :func:`probly.method.g_vbll.find_g_vbll_layer`.

    Args:
        model: The model to search, typically the result of :func:`vbll`.

    Returns:
        The first VBLL layer in forward DFS order.

    Raises:
        ValueError: If the model contains no VBLL layer.
    """
    from probly.layers.torch import HetVBLLLayer, TVBLLLayer, VBLLLayer  # noqa: PLC0415

    return find_layer(model, (VBLLLayer, TVBLLLayer, HetVBLLLayer))


@flexdispatch
def compute_vbll_categorical_sample(sample: Sample[Any]) -> CategoricalDistributionSample[Any]:
    """Convert a sample of VBLL logits to a categorical distribution sample."""
    msg = f"compute_vbll_categorical_sample not implemented for type {type(sample)}."
    raise NotImplementedError(msg)


@representer.register(VBLLPredictor)
class VBLLRepresenter[**In, Out](Representer[Any, In, Out, CategoricalDistributionSample[Any]]):
    """Representer that turns the VBLL logit Gaussian into categorical samples.

    A single network forward yields the closed-form Gaussian over logits;
    ``num_samples`` logit samples are then drawn from that Gaussian and passed
    through the softmax, producing a :class:`CategoricalDistributionSample`
    (the Monte-Carlo softmax predictive of :cite:`harrisonVariationalBayesian2024`).
    Because VBLL is Bayesian over the last-layer weights, the resulting sample
    carries both aleatoric and epistemic uncertainty and routes to the default
    second-order entropy decomposition.
    """

    num_samples: int

    def __init__(
        self,
        predictor: Predictor[In, Out],
        num_samples: int = 10,
        *args: In.args,
        **kwargs: In.kwargs,
    ) -> None:
        """Initialize the VBLL representer.

        Args:
            predictor: The VBLL predictor to sample from.
            num_samples: Number of logit samples drawn from the predictive Gaussian.
            *args: Additional positional arguments forwarded to the base class.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(predictor, *args, **kwargs)
        self.num_samples = num_samples

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Predict the closed-form logit Gaussian from the VBLL predictor."""
        return predict(self.predictor, *args, **kwargs)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> CategoricalDistributionSample[Any]:
        """Sample logits from the predictive Gaussian and softmax them into categoricals."""
        distribution = self._predict(*args, **kwargs)
        sampled_logits = distribution.sample(self.num_samples)  # ty:ignore[unresolved-attribute]
        return compute_vbll_categorical_sample(sampled_logits)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@VBLLPredictor.register_factory
def vbll[**In, Out: GaussianDistribution](
    base: Predictor[In, Out],
    variant: str = "discriminative",
    parameterization: str = "dense",
    prior_scale: float = 1.0,
    noise_init: float = math.exp(-1.0),
    cov_rank: int = 3,
    wishart_scale: float = 1.0,
    dof: float = 2.0,
    noise_prior_scale: float = 0.01,
) -> VBLLPredictor[In, Out]:
    """Wrap a model with a Variational Bayesian Last Layer (VBLL).

    Replaces the model's last ``nn.Linear`` with a variational Bayesian last
    layer and emits, in closed form, a Gaussian over the network outputs based on
    :cite:`harrisonVariationalBayesian2024`. A trailing softmax (if any) is
    removed, since the layer outputs logits.

    Three discriminative ``variant`` s are available, all producing a Gaussian
    over logits and sharing the same predict/representer pipeline:

    - ``"discriminative"``: the standard VBLL classifier (:class:`VBLLLayer`),
      also usable for regression.
    - ``"student_t"``: additionally infers the noise variance via a Gamma
      posterior, giving a Student-t marginal (:class:`TVBLLLayer`).
    - ``"heteroscedastic"``: input-dependent noise via a second weight posterior
      (:class:`HetVBLLLayer`).

    The returned predictor's :func:`predict` yields the closed-form
    :class:`GaussianDistribution` (the regression predictive, or the logit
    Gaussian for classification). For classification, use the registered
    :class:`VBLLRepresenter` to obtain a categorical sample and an
    aleatoric/epistemic decomposition.

    Args:
        base: The model to wrap.
        variant: The last-layer variant, one of ``"discriminative"``,
            ``"student_t"`` or ``"heteroscedastic"``. Defaults to ``"discriminative"``.
        parameterization: Posterior covariance parametrization. The
            ``"discriminative"`` variant supports ``"diagonal"``, ``"dense"`` and
            ``"lowrank"``; the other variants support ``"diagonal"`` and ``"dense"``.
            Defaults to ``"dense"``.
        prior_scale: Scale of the isotropic prior covariance. Defaults to ``1.0``.
        noise_init: Median of the random initial per-output noise standard
            deviation (``"discriminative"`` variant only). Defaults to
            ``exp(-1)``, matching the reference initialization.
        cov_rank: Rank of the low-rank covariance factor (only used when
            ``parameterization="lowrank"``). Defaults to ``3``.
        wishart_scale: Scale of the Wishart/Gamma prior on the noise precision
            (``"discriminative"`` and ``"student_t"`` variants). Defaults to ``1.0``.
        dof: Degrees of freedom of the Wishart/Gamma prior on the noise precision
            (``"discriminative"`` and ``"student_t"`` variants; must be > 1 for
            ``"student_t"``). Defaults to ``2.0``.
        noise_prior_scale: Scale of the prior on the input-dependent noise weights
            (``"heteroscedastic"`` variant only). Defaults to ``0.01``.

    Returns:
        A ``VBLLPredictor`` whose ``predict(...)`` returns a
        ``GaussianDistribution`` over the outputs.
    """
    if variant not in ("discriminative", "student_t", "heteroscedastic"):
        msg = f"variant must be one of 'discriminative', 'student_t' or 'heteroscedastic', but got {variant!r} instead."
        raise ValueError(msg)
    if parameterization not in ("diagonal", "dense", "lowrank"):
        msg = f"parameterization must be one of 'diagonal', 'dense' or 'lowrank', but got {parameterization!r} instead."
        raise ValueError(msg)
    return traverse(
        base,
        nn_compose(vbll_traverser),
        init={
            CLONE: True,
            TRAVERSE_REVERSED: True,
            LAST_LAYER: True,
            VARIANT: variant,
            PARAMETERIZATION: parameterization,
            PRIOR_SCALE: prior_scale,
            NOISE_INIT: noise_init,
            COV_RANK: cov_rank,
            WISHART_SCALE: wishart_scale,
            DOF: dof,
            NOISE_PRIOR_SCALE: noise_prior_scale,
        },
    )


@predict.register(VBLLPredictor)
def _[**In](
    predictor: VBLLPredictor[In, GaussianDistribution], *args: In.args, **kwargs: In.kwargs
) -> GaussianDistribution:
    """Predict the closed-form predictive Gaussian for a VBLL predictor."""
    mean, variance = predict_raw(predictor, *args, **kwargs)
    return create_gaussian_distribution(mean, variance)
