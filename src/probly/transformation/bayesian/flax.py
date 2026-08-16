"""Flax Bayesian implementation."""

from __future__ import annotations

from flax.nnx import Linear, Rngs, rnglib

from probly.layers.flax import BayesLinear

from ._common import register


def replace_flax_bayesian_linear(
    obj: Linear,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
    rngs: rnglib.Rngs | rnglib.RngStream | int,
    rng_collection: str = "bayesian",
) -> BayesLinear:
    """Replace a given layer by a BayesLinear layer :cite:`blundellWeightUncertainty2015`."""
    if isinstance(rngs, int):
        rngs = Rngs(rngs)
    return BayesLinear(
        base_layer=obj,
        use_base_weights=use_base_weights,
        posterior_std=posterior_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        rng_collection=rng_collection,
        rngs=rngs,
    )


register(Linear, replace_flax_bayesian_linear)
