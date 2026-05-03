"""Flax implementation of HetNets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax

from probly.layers.flax import HeteroscedasticLayer

from ._common import (
    IS_PARAMETER_EFFICIENT,
    LAST_LAYER,
    NUM_FACTORS,
    RNGS,
    TEMPERATURE,
    het_net_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


# Identity-matched callables that are equivalent to a torch ``nn.Softmax`` tail
# and that the het_net transformation should strip from a trailing position
# (the HeteroscedasticLayer applies its own activation internally, so leaving a
# softmax behind would double-activate the output).
_SOFTMAX_TAILS: frozenset[object] = frozenset(
    {
        jax.nn.softmax,
        jax.nn.log_softmax,
        nnx.softmax,
        nnx.log_softmax,
    }
)


@het_net_traverser.register(nnx.Module)
def skip_layer(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    """Traverser for unchanged flax layers."""
    return obj, state


@het_net_traverser.register(nnx.Linear)
def drop_in_place_het_layer(obj: nnx.Linear, state: State) -> tuple[nnx.Module, State]:
    """Replace the last linear layer with a HeteroscedasticLayer."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return HeteroscedasticLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            num_factors=state[NUM_FACTORS],
            temperature=state[TEMPERATURE],
            is_parameter_efficient=state[IS_PARAMETER_EFFICIENT],
            rngs=state[RNGS],
        ), state
    return obj, state


@het_net_traverser.register(nnx.Sequential)
def strip_trailing_softmax(obj: nnx.Sequential, state: State) -> tuple[nnx.Module, State]:
    """Strip a trailing softmax-like callable from the end of a ``Sequential``.

    Mirrors the torch handler that replaces a trailing ``nn.Softmax`` with
    ``nn.Identity`` -- the HeteroscedasticLayer applies its own activation, so a
    bare ``jax.nn.softmax`` (or sibling) at the tail would double-activate.

    Matching is by identity against a small allowlist of known callables; an
    arbitrary callable in the tail position is left untouched.
    """
    layers = list(obj.layers)
    if layers and layers[-1] in _SOFTMAX_TAILS:
        return nnx.Sequential(*layers[:-1]), state
    return obj, state
