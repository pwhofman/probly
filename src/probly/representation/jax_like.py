"""Protocol for ndarray-like objects."""

from __future__ import annotations

from abc import ABC

import jax

from probly.representation.array_like import ArrayLike


class JaxLikeImplementation[DT](ArrayLike[DT], ABC):
    """Protocol for array-like objects that behave like JAX arrays."""


JaxLikeImplementation.register(jax.Array)
