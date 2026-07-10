"""Shared Generative Variational Bayesian Last Layer (G-VBLL) implementation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from probly.predictor import (
    LogitDistributionPredictor,
    Predictor,
    predict,
    predict_raw,
)
from probly.representation.distribution import (
    CategoricalDistribution,
    create_categorical_distribution_from_logits,
)
from probly.transformation.transformation import predictor_transformation
from probly.traverse_nn import find_layer, nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

g_vbll_traverser = flexdispatch_traverser[object](name="g_vbll_traverser")

LAST_LAYER = GlobalVariable[bool]("LAST_LAYER", "Whether the current layer is the last layer of the model.")
PRIOR_SCALE = GlobalVariable[float]("PRIOR_SCALE", "The scale of the isotropic prior covariance.")
NOISE_INIT = GlobalVariable[float]("NOISE_INIT", "The initial per-output noise standard deviation.")
WISHART_SCALE = GlobalVariable[float]("WISHART_SCALE", "The scale of the Wishart prior on the noise precision.")
DOF = GlobalVariable[float]("DOF", "The degrees of freedom of the Wishart prior on the noise precision.")


@runtime_checkable
class GVBLLPredictor[**In, Out: CategoricalDistribution](LogitDistributionPredictor[In, Out], Protocol):
    """A generative variational Bayesian last layer (G-VBLL) classifier.

    Unlike the discriminative :class:`~probly.method.vbll.VBLLPredictor`, G-VBLL
    models a per-class Gaussian density in feature space.  Its :func:`predict`
    returns a deterministic :class:`CategoricalDistribution` whose logits are the
    class-conditional log-densities; the softmax is distance-aware and reverts to
    the uniform distribution far from every class.
    """


def find_g_vbll_layer(model: object) -> Any:  # noqa: ANN401, avoids importing the torch layer type eagerly
    """Return the generative variational Bayesian last layer of a transformed predictor.

    Convenience wrapper around :func:`probly.traverse_nn.find_layer` that matches
    the :class:`~probly.layers.torch.GVBLLLayer`, e.g. to pass the layer to
    :func:`probly.train.vbll.vbll_loss` or to attach hooks to it. For
    discriminative VBLL models use :func:`probly.method.vbll.find_vbll_layer`.

    Args:
        model: The model to search, typically the result of :func:`g_vbll`.

    Returns:
        The first G-VBLL layer in forward DFS order.

    Raises:
        ValueError: If the model contains no G-VBLL layer.
    """
    from probly.layers.torch import GVBLLLayer  # noqa: PLC0415

    return find_layer(model, GVBLLLayer)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@GVBLLPredictor.register_factory
def g_vbll[**In, Out: CategoricalDistribution](
    base: Predictor[In, Out],
    prior_scale: float = 1.0,
    noise_init: float = 1.0,
    wishart_scale: float = 1.0,
    dof: float = 1.0,
) -> GVBLLPredictor[In, Out]:
    """Wrap a model with a Generative Variational Bayesian Last Layer (G-VBLL).

    Replaces the model's last ``nn.Linear`` with a
    :class:`~probly.layers.torch.GVBLLLayer` that models a per-class Gaussian
    density in feature space based on :cite:`harrisonVariationalBayesian2024`.  A
    trailing softmax (if any) is removed, since the layer outputs class-conditional
    log-densities (logits).

    The returned predictor's :func:`predict` yields a deterministic
    :class:`CategoricalDistribution`.  Because each class density decays
    quadratically away from its mean, the predictive is distance-aware -- a useful
    property for out-of-distribution detection.  The layer is fit with the
    generative ELBO exposed by :func:`probly.train.vbll.torch.g_vbll_loss`.

    Args:
        base: The model to wrap.
        prior_scale: Scale of the isotropic Gaussian prior on the class means.
            Defaults to ``1.0``.
        noise_init: Median of the random initial shared feature-noise standard
            deviation. The default of ``1.0`` matches the reference initialization.
        wishart_scale: Scale of the Wishart prior on the noise precision. Defaults to ``1.0``.
        dof: Degrees of freedom of the Wishart prior on the noise precision. Defaults to ``1.0``.

    Returns:
        A ``GVBLLPredictor`` whose ``predict(...)`` returns a
        ``CategoricalDistribution`` over the classes.
    """
    return traverse(
        base,
        nn_compose(g_vbll_traverser),
        init={
            CLONE: True,
            TRAVERSE_REVERSED: True,
            LAST_LAYER: True,
            PRIOR_SCALE: prior_scale,
            NOISE_INIT: noise_init,
            WISHART_SCALE: wishart_scale,
            DOF: dof,
        },
    )


@predict.register(GVBLLPredictor)
def _[**In](
    predictor: GVBLLPredictor[In, CategoricalDistribution], *args: In.args, **kwargs: In.kwargs
) -> CategoricalDistribution:
    """Predict the deterministic categorical distribution for a G-VBLL predictor."""
    return create_categorical_distribution_from_logits(predict_raw(predictor, *args, **kwargs))
