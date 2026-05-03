"""Flax SNGP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp

from probly.layers.flax import SNGPLayer, SpectralNormWithMultiplier
from probly.representation.distribution._common import create_categorical_distribution_from_logits
from probly.representation.distribution.jax_categorical import JaxCategoricalDistributionSample
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
#
# Note: matching here is by object identity. Lambdas, ``functools.partial``
# wrappers, or otherwise-rewrapped softmax callables are not detected and will
# remain in place. Users in such cases should remove the trailing softmax
# manually before applying ``sngp``.
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


@compute_categorical_sample_from_logits.register(JaxArraySample)
def jax_compute_categorical_sample_from_logits(
    sample: JaxArraySample,
) -> JaxCategoricalDistributionSample:
    """Convert a ``JaxArraySample`` of SNGP logits to a categorical distribution sample."""
    array = sample.array
    sample_axis = sample.sample_axis
    if array.ndim >= 3 and sample_axis == 0:
        array = jnp.swapaxes(array, 0, 1)
        sample_axis = 1

    categorical_dist = create_categorical_distribution_from_logits(array)
    return JaxCategoricalDistributionSample(array=categorical_dist, sample_axis=sample_axis)  # ty:ignore[invalid-argument-type]
