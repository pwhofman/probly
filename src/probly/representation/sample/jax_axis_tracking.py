"""Axis-tracking for PyTorch tensors."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.representation.sample.axis_tracking import ArrayIndex, convert_idx


@convert_idx.register(jax.Array)
def _convert_jax_array_idx(idx: jax.Array) -> ArrayIndex | bool | int:
    if idx.ndim == 0:
        if idx.dtype == jnp.bool_:
            return bool(idx)
        return 0
    return ArrayIndex(index=idx, ndim=idx.ndim, is_boolean=idx.dtype == jnp.bool_)
