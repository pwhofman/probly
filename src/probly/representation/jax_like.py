"""Protocol for ndarray-like objects."""

from __future__ import annotations

from abc import ABC
from typing import Any

import jax

from probly.representation.array_like import ArrayLike


class JaxArrayLikeImplementation(ArrayLike[Any], ABC):
    """Protocol for array-like objects that behave like JAX arrays."""


JaxArrayLikeImplementation.register(jax.Array)
