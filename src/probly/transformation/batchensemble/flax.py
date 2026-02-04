"""Flax.nnx BatchEnsemble implementation."""

from __future__ import annotations

from flax import nnx

from probly.layers.flax import BatchEnsembleConv, BatchEnsembleLinear

from .common import register


def replace_flax_batchensemble_linear(
    obj: nnx.Linear,
    num_members: int,
    use_base_weights: bool,
    s_mean: float,
    s_std: float,
    r_mean: float,
    r_std: float,
    rngs: nnx.Rngs | int,
) -> BatchEnsembleLinear:
    """Replace a given layer by a BatchEnsembleLinear layer."""
    return BatchEnsembleLinear(
        base_layer=obj,
        rngs=rngs,
        num_members=num_members,
        use_base_weights=use_base_weights,
        s_mean=s_mean,
        s_std=s_std,
        r_mean=r_mean,
        r_std=r_std,
    )


def replace_flax_batchensemble_conv(
    obj: nnx.Conv,
    num_members: int,
    use_base_weights: bool,
    s_mean: float,
    s_std: float,
    r_mean: float,
    r_std: float,
    rngs: nnx.Rngs | int,
) -> BatchEnsembleConv:
    """Replace a given layer by a BatchEnsembleConv layer."""
    return BatchEnsembleConv(
        base_layer=obj,
        rngs=rngs,
        num_members=num_members,
        use_base_weights=use_base_weights,
        s_mean=s_mean,
        s_std=s_std,
        r_mean=r_mean,
        r_std=r_std,
    )


register(nnx.Linear, replace_flax_batchensemble_linear)
register(nnx.Conv, replace_flax_batchensemble_conv)
