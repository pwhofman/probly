"""Shared normal-inverse-gamma head transformation implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, TRAVERSE_REVERSED, GlobalVariable, flexdispatch_traverser, traverse

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs, RngStream

    from flextype.isinstance import LazyType
    from probly.predictor import Predictor
    from pytraverse.composition import RegisteredLooseTraverser

REPLACED_LAST_LINEAR = GlobalVariable[bool](
    "REPLACED_LAST_LINEAR",
    "Whether the last linear layer has been replaced with a NormalInverseGammaLinear layer.",
    default=False,
)

type RNG = int | Rngs | RngStream

RNGS = GlobalVariable[RNG](
    "NIG_RNGS",
    "rngs used to initialize the flax NormalInverseGammaLinear replacement layer.",
    default=1,
)

normal_inverse_gamma_head_traverser = flexdispatch_traverser[object](name="normal_inverse_gamma_head_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be replaced by a normal inverse gamma layer."""
    normal_inverse_gamma_head_traverser.register(
        cls=cls, traverser=traverser, skip_if=lambda s: s[REPLACED_LAST_LINEAR]
    )


def normal_inverse_gamma_head[T: Predictor](base: T, rngs: RNG = 1) -> T:
    """Replace the final linear layer with a Normal-Inverse-Gamma head.

    Args:
        base: Predictor, The base model to be transformed.
        rngs: Optional rngs for the flax NormalInverseGammaLinear initialization
            (types: ``rnglib.Rngs | rnglib.RngStream | int``). Ignored by the torch
            backend. Default is ``1``.

    Returns:
        Predictor, The transformed predictor.
    """
    return traverse(
        base,
        nn_compose(normal_inverse_gamma_head_traverser),
        init={TRAVERSE_REVERSED: True, CLONE: True, RNGS: rngs},
    )
