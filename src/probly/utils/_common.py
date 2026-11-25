"""Utility functions dispatching on the backend of their inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.lazy_types import NativeArray


@flexdispatch
def elementwise_entropy[A: NativeArray](p: A) -> A:
    """Compute the entropy contribution ``-p * log(p)`` of every probability.

    Args:
        p: Probabilities of any shape as a numpy array, a torch tensor or a jax array.

    Returns:
        Contributions of the shape of ``p`` in nats, with ``0 * log(0)`` and non-positive entries counted as zero.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``p``.
    """
    msg = f"No elementwise entropy implementation registered for type {type(p)}"
    raise NotImplementedError(msg)


@flexdispatch
def entropy(p: object) -> object:
    """Compute the Shannon entropy of probability vectors along their last axis.

    Args:
        p: Probabilities of shape ``(..., num_classes)`` as a numpy array, a torch tensor or a jax array.

    Returns:
        Entropies of shape ``(...)`` in nats, with ``0 * log(0)`` counted as zero.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``p``.
    """
    msg = f"No entropy implementation registered for type {type(p)}"
    raise NotImplementedError(msg)


@flexdispatch
def intersection_probability(lower: object, upper: object) -> object:
    """Reduce probability intervals to their intersection probability :cite:`wangCreINNsCredalSet2025`.

    Args:
        lower: Lower bounds of shape ``(..., num_classes)`` as a torch tensor or a jax array.
        upper: Upper bounds of the same type and shape as ``lower``.

    Returns:
        Probability vectors of shape ``(..., num_classes)``.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``lower``.
    """
    msg = f"No intersection probability implementation registered for type {type(lower)}"
    raise NotImplementedError(msg)
