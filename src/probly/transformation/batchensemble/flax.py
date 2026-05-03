"""Flax.nnx BatchEnsemble implementation."""

from __future__ import annotations

from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp

from probly.layers.flax import BatchEnsembleConv, BatchEnsembleLinear
from probly.predictor import predict
from probly.representation.sample.jax import JaxArraySample

from ._common import BatchEnsemblePredictor, _attach_num_members, register


def tile_inputs(x: jax.Array, num_members: int) -> jax.Array:
    """Tile the leading batch dim by ``num_members`` for a BatchEnsemble forward pass."""
    return jnp.tile(x, (num_members,) + (1,) * (x.ndim - 1))


@_attach_num_members.register(nnx.Module)
def _(model: nnx.Module, num_members: int) -> None:
    """Attach ``num_members`` as a plain attribute (preserved through ``nnx.split``)."""
    model.num_members = num_members  # ty: ignore[unresolved-attribute]


@predict.register(BatchEnsemblePredictor)
def predict_batchensemble(
    predictor: BatchEnsemblePredictor,
    x: jax.Array,
) -> JaxArraySample:
    """Run a BatchEnsemble predictor and return a :class:`JaxArraySample` over members.

    Tiles the user's ``[B, ...]`` input by ``num_members``, runs the model on the
    ``[E*B, ...]`` array, and reshapes the output to ``[E, B, ...]`` with
    ``sample_axis=0``.
    """
    num_members = int(predictor.num_members)
    b = x.shape[0]
    raw = predictor(tile_inputs(x, num_members))
    out = raw.reshape(num_members, b, *raw.shape[1:])
    return JaxArraySample(array=out, sample_axis=0)


def replace_flax_batchensemble_linear(
    obj: nnx.Linear,
    num_members: int,
    use_base_weights: bool,
    init: Literal["random_sign", "normal"],
    r_mean: float,
    r_std: float,
    s_mean: float,
    s_std: float,
    rngs: nnx.Rngs | int,
) -> BatchEnsembleLinear:
    """Replace a given layer by a BatchEnsembleLinear layer."""
    return BatchEnsembleLinear(
        base_layer=obj,
        rngs=rngs,
        num_members=num_members,
        use_base_weights=use_base_weights,
        init=init,
        r_mean=r_mean,
        r_std=r_std,
        s_mean=s_mean,
        s_std=s_std,
    )


def replace_flax_batchensemble_conv(
    obj: nnx.Conv,
    num_members: int,
    use_base_weights: bool,
    init: Literal["random_sign", "normal"],
    r_mean: float,
    r_std: float,
    s_mean: float,
    s_std: float,
    rngs: nnx.Rngs | int,
) -> BatchEnsembleConv:
    """Replace a given layer by a BatchEnsembleConv layer."""
    return BatchEnsembleConv(
        base_layer=obj,
        rngs=rngs,
        num_members=num_members,
        use_base_weights=use_base_weights,
        init=init,
        r_mean=r_mean,
        r_std=r_std,
        s_mean=s_mean,
        s_std=s_std,
    )


register(nnx.Linear, replace_flax_batchensemble_linear)
register(nnx.Conv, replace_flax_batchensemble_conv)
