"""Flax SNGP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from probly.layers.flax import SNGPLayer, SpectralNormWithMultiplier
from probly.representation.distribution import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
    create_gaussian_distribution,
)
from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution
from probly.representation.sample.array import ArraySample
from probly.representation.sample.jax import JaxArraySample

from ._common import (
    EPS,
    LAST_LAYER,
    MOMENTUM,
    N_POWER_ITERATIONS,
    NORM_MULTIPLIER,
    NUM_INDUCING,
    RIDGE_PENALTY,
    RNGS,
    compute_categorical_sample_from_logits,
    sngp_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State

# Identity-matched callables that the spectral-norm transformation should strip
# from a trailing position in a Sequential. The SNGPLayer returns logits-and-
# variance directly, so a bare softmax callable on the tail would corrupt the
# expected (logits, variance) signature.
_SOFTMAX_TAILS: frozenset[object] = frozenset(
    {
        jax.nn.softmax,
        jax.nn.log_softmax,
        nnx.softmax,
        nnx.log_softmax,
    },
)


def _resolve_kernel_name(obj: nnx.Module, requested: str) -> str:
    """Return the attribute name to spectrally normalize on ``obj``.

    Defaults from the torch convention (``"weight"``) are mapped to flax's
    ``"kernel"`` when the requested attribute does not exist on the layer.
    """
    if hasattr(obj, requested):
        return requested
    if requested == "weight" and hasattr(obj, "kernel"):
        return "kernel"
    return requested


@sngp_traverser.register(nnx.Module)
def skip_layer(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    """Default flax SNGP handler. No-op for layers without a kernel to wrap."""
    return obj, state


@sngp_traverser.register(nnx.Linear)
def replace_linear_with_sngp(obj: nnx.Linear, state: State) -> tuple[nnx.Module, State]:
    """Replace the last ``nnx.Linear`` with an ``SNGPLayer`` and wrap earlier ones."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return SNGPLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            num_inducing=state[NUM_INDUCING],
            ridge_penalty=state[RIDGE_PENALTY],
            momentum=state[MOMENTUM],
            rngs=state[RNGS],
        ), state
    return SpectralNormWithMultiplier(
        module=obj,
        name=_resolve_kernel_name(obj, "kernel"),
        n_power_iterations=state[N_POWER_ITERATIONS],
        norm_multiplier=state[NORM_MULTIPLIER],
        eps=state[EPS],
        rngs=state[RNGS],
    ), state


@sngp_traverser.register(nnx.Conv)
def wrap_conv_with_spectral_norm(obj: nnx.Conv, state: State) -> tuple[nnx.Module, State]:
    """Wrap an ``nnx.Conv`` with spectral-norm parameterization."""
    return SpectralNormWithMultiplier(
        module=obj,
        name=_resolve_kernel_name(obj, "kernel"),
        n_power_iterations=state[N_POWER_ITERATIONS],
        norm_multiplier=state[NORM_MULTIPLIER],
        eps=state[EPS],
        rngs=state[RNGS],
    ), state


@sngp_traverser.register(nnx.Sequential)
def strip_trailing_softmax(obj: nnx.Sequential, state: State) -> tuple[nnx.Module, State]:
    """Strip a trailing softmax-like callable from a ``Sequential``.

    Mirrors the torch handler that replaces a trailing ``nn.Softmax`` with
    ``nn.Identity``. The SNGPLayer returns ``(logits, variance)`` directly, so a
    bare softmax callable in the tail position would corrupt the call signature.
    """
    layers = list(obj.layers)
    if layers and layers[-1] in _SOFTMAX_TAILS:
        return nnx.Sequential(*layers[:-1]), state
    return obj, state


@create_gaussian_distribution.register(jax.Array)
def _create_array_gaussian_distribution_from_jax(
    mean: jax.Array,
    var: jax.Array | None = None,
) -> ArrayGaussianDistribution:
    """Build an ``ArrayGaussianDistribution`` from JAX-array mean/variance.

    The SNGP flax forward returns a ``(logits, variance)`` tuple of ``jax.Array``s
    that the predictor's ``predict`` plumbing routes to ``create_gaussian_distribution``.
    Falling through to the numpy registration would not match because
    ``jax.Array`` is not an ``np.ndarray`` subclass.
    """
    mean_np = np.asarray(mean)
    if var is None:
        if mean_np.shape[-1] != 2:
            msg = "If var is not provided, mean must have shape (..., 2) where the last axis contains [mean, var]"
            raise ValueError(msg)
        return ArrayGaussianDistribution(mean=mean_np[..., 0], var=mean_np[..., 1])
    return ArrayGaussianDistribution(mean=mean_np, var=np.asarray(var))


def _categorical_sample_from_array(
    array: np.ndarray | jax.Array,
    sample_axis: int,
) -> ArrayCategoricalDistributionSample:
    """Build an ``ArrayCategoricalDistributionSample`` from a logits array.

    Mirrors the torch handler: when the leading axis is the sample axis on a
    ``ndim >= 3`` array we transpose so the categorical sample axis lands on
    axis 1, then apply softmax along the trailing class axis.
    """
    if array.ndim >= 3 and sample_axis == 0:
        array = jnp.swapaxes(array, 0, 1) if isinstance(array, jax.Array) else np.swapaxes(array, 0, 1)
        sample_axis = 1

    if isinstance(array, jax.Array):
        probabilities = np.asarray(jax.nn.softmax(array, axis=-1))
    else:
        shifted = array - np.max(array, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / np.sum(exp, axis=-1, keepdims=True)

    distribution = ArrayCategoricalDistribution(unnormalized_probabilities=probabilities)
    return ArrayCategoricalDistributionSample(array=distribution, sample_axis=sample_axis)


@compute_categorical_sample_from_logits.register(JaxArraySample)
def jax_compute_categorical_sample_from_logits(
    sample: JaxArraySample,
) -> ArrayCategoricalDistributionSample:
    """Convert a ``JaxArraySample`` of SNGP logits to a categorical distribution sample.

    The result is a numpy-backed ``ArrayCategoricalDistributionSample`` so it
    flows through the existing array-based decomposition dispatch.
    """
    return _categorical_sample_from_array(sample.array, sample.sample_axis)


@compute_categorical_sample_from_logits.register(ArraySample)
def array_compute_categorical_sample_from_logits(
    sample: ArraySample,
) -> ArrayCategoricalDistributionSample:
    """Convert a numpy-backed ``ArraySample`` of SNGP logits to a categorical sample.

    The flax SNGP path returns Gaussian-distribution samples drawn through the
    numpy-backed ``ArrayGaussianDistribution``; that produces an ``ArraySample``
    that this handler routes into the same categorical-distribution sample type.
    """
    return _categorical_sample_from_array(sample.array, sample.sample_axis)
