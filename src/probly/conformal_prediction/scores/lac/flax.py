"""Flax for LAC."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt


def lac_jax(probs: Any) -> npt.NDArray[np.floating]:
    """Compute APS scores for JAX arrays."""
    lac_scores = 1.0 - probs
    return lac_scores  # shape: (n_samples, n_classes)


from .common import register

register("jax.numpy.ndarray", lac_jax)
