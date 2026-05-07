"""Flax normal-inverse-gamma head implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from probly.layers.flax import NormalInverseGammaLinear
from probly.transformation.normal_inverse_gamma_head._common import REPLACED_LAST_LINEAR, RNGS, register

if TYPE_CHECKING:
    from pytraverse import State, TraverserResult


def replace_last_flax_nig(obj: nnx.Linear, state: State) -> TraverserResult:
    """Register a class to be replaced by the NormalInverseGammaLinear layer based on :cite:`aminiDeepEvidential2020`.

    This layer outputs the parameters of a Normal Inverse Gamma distribution, which is central to evidential
    regression.
    """
    state[REPLACED_LAST_LINEAR] = True
    return NormalInverseGammaLinear(
        obj.in_features,
        obj.out_features,
        rngs=state[RNGS],
        use_bias=obj.use_bias,
        param_dtype=obj.param_dtype,
    ), state


register(nnx.Linear, replace_last_flax_nig)
