"""JAX implementations of point deciders."""

from __future__ import annotations

import jax

from ._common import point_from_mean


@point_from_mean.register(jax.Array)
def _(prediction: jax.Array) -> jax.Array:
    return prediction
