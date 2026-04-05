"""Protocol for ndarray-like objects."""

from __future__ import annotations

from typing import Any, Protocol

import jax

from lazy_dispatch import ProtocolRegistry
from probly.representation.array_like import ArrayLike


class JaxArrayLike(ArrayLike[Any], ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for array-like objects that behave like JAX arrays."""


JaxArrayLike.register(jax.Array)
