"""Backend-agnostic probability extraction helpers for credal set plotting.

These helpers live in their own module so that both the public entry point
(``plot.py``) and the per-shape drawing modules (``_binary``, ``_ternary``,
``_spider``) can import them without creating an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from flextype import flexdispatch
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArraySingletonCredalSet,
)

if TYPE_CHECKING:
    from probly.representation.credal_set._common import CategoricalCredalSet


def _to_numpy(val: object) -> np.ndarray:
    """Safely detach tensors to numpy arrays, or pass arrays through."""
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy()  # ty: ignore[call-non-callable]
    return np.asarray(val)


def _flatten_batch[T: CategoricalCredalSet](data: T) -> T:
    """Flatten batch dimensions if the credal set supports it (Torch or Array)."""
    reshape_fn = getattr(data, "reshape", None)
    if callable(reshape_fn):
        return reshape_fn(-1)

    msg = (
        f"Input of type {type(data).__name__} is not a supported batched credal set. "
        "Plotting requires credal sets from standard backends (Torch or Array) that implement '.reshape()'."
    )
    raise TypeError(msg)


@flexdispatch
def _get_unnormalized_probabilities(data: object) -> np.ndarray:
    """Extract the unnormalized probability array backing a credal set.

    The shape of the returned array depends on the credal set type: singleton
    and distance-based sets return one distribution per batch element, while
    vertex-based sets (discrete, convex) return all vertices.

    Args:
        data: The credal set to extract probabilities from.

    Returns:
        The unnormalized probabilities as a NumPy array.

    Raises:
        NotImplementedError: If no handler is registered for the given type.
    """
    msg = f"Cannot extract probabilities from {type(data).__name__}"
    raise NotImplementedError(msg)


@_get_unnormalized_probabilities.register(ArraySingletonCredalSet | ArrayDiscreteCredalSet | ArrayConvexCredalSet)
def _array_probabilities(data: ArraySingletonCredalSet | ArrayDiscreteCredalSet | ArrayConvexCredalSet) -> np.ndarray:
    return _to_numpy(data.array.unnormalized_probabilities)


@_get_unnormalized_probabilities.register(ArrayDistanceBasedCredalSet)
def _nominal_probabilities(data: ArrayDistanceBasedCredalSet) -> np.ndarray:
    return _to_numpy(data.nominal.unnormalized_probabilities)
