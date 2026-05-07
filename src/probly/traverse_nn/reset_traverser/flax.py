"""Flax implementation of the reset traverser."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp

from ._common import RNGS, reset_traverser

if TYPE_CHECKING:
    from flax.nnx import rnglib

    from pytraverse.core import State


def _coerce_rngs(rngs: int | nnx.Rngs | rnglib.RngStream) -> nnx.Rngs:
    """Return an :class:`nnx.Rngs` regardless of how the rng source was provided."""
    if isinstance(rngs, nnx.Rngs):
        return rngs
    return nnx.Rngs(rngs)


@reset_traverser.register(cls=nnx.Module)
def skip_module(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    """Default flax reset handler. No-op for modules without trainable params.

    Mirrors the torch behavior of only resetting layers that explicitly support it.
    """
    return obj, state


def _resolve_rngs(state: State) -> nnx.Rngs:
    """Coerce the state-provided rng source into a persistent :class:`nnx.Rngs`.

    Stores the coerced ``Rngs`` back into the state so subsequent layer resets advance
    the same stream rather than sampling deterministically from the same seed.
    """
    rngs = state[RNGS]
    if isinstance(rngs, nnx.Rngs):
        return rngs
    coerced = _coerce_rngs(rngs)
    state[RNGS] = coerced
    return coerced


@reset_traverser.register(cls=nnx.Linear)
def reset_linear(obj: nnx.Linear, state: State) -> tuple[nnx.Module, State]:
    """Reset the kernel (and bias if present) of an :class:`nnx.Linear` layer."""
    rngs = _resolve_rngs(state)
    kernel_init = jax.nn.initializers.lecun_normal()
    obj.kernel.value = kernel_init(rngs.params(), obj.kernel.value.shape, obj.param_dtype)
    if obj.use_bias and obj.bias is not None:
        obj.bias.value = jnp.zeros(obj.bias.value.shape, dtype=obj.param_dtype)
    return obj, state


@reset_traverser.register(cls=nnx.Conv)
def reset_conv(obj: nnx.Conv, state: State) -> tuple[nnx.Module, State]:
    """Reset the kernel (and bias if present) of an :class:`nnx.Conv` layer."""
    rngs = _resolve_rngs(state)
    kernel_init = jax.nn.initializers.lecun_normal()
    obj.kernel.value = kernel_init(rngs.params(), obj.kernel.value.shape, obj.param_dtype)
    if obj.use_bias and obj.bias is not None:
        obj.bias.value = jnp.zeros(obj.bias.value.shape, dtype=obj.param_dtype)
    return obj, state
