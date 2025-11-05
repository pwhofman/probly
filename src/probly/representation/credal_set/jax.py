"""Torch credal set implementation."""

from __future__ import annotations

import jax

from probly.representation.credal_set.credal_set import CredalSet, credal_set_from_sample
from probly.representation.sampling.jax_sample import JaxArraySample


@credal_set_from_sample.register(JaxArraySample)
class JaxArrayCredalSet(CredalSet[jax.Array]):
    """A credal set implementation for torch tensors."""

    def __init__(self, sample: JaxArraySample) -> None:
        """Initialize the torch tensor credal set."""
        self.array = sample.array

    def lower(self) -> jax.Array:
        """Compute the lower envelope of the credal set."""
        return self.array.min(axis=1)

    def upper(self) -> jax.Array:
        """Compute the upper envelope of the credal set."""
        return self.array.max(axis=1)
