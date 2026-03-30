"""Utility functions regarding sets."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def powerset(iterable: Iterable[int]) -> list[tuple[()]]:
    """Generate the power set of a given iterable.

    Args:
        iterable: An iterable of integers.

    Returns:
        Power set of the given iterable.

    """
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))


def capacity(q: np.ndarray, a: Iterable[int]) -> np.ndarray:
    """Compute the capacity of set q given set a.

    Args:
        q: Array of shape (n_instances, n_samples, n_classes).
        a: Indices indicating subset of classes.

    Returns:
        Capacity of q given a of shape (n_instances,).

    """
    selected_sum = np.sum(q[:, :, a], axis=2)  # ty: ignore[invalid-argument-type]
    min_capacity = np.min(selected_sum, axis=1)
    return min_capacity


def moebius(q: np.ndarray, a: Iterable[int]) -> np.ndarray:
    """Compute the Moebius function of a set q given a set a.

    Args:
        q: Array of shape (n_instances, n_samples, n_classes).
        a: Indices indicating subset of classes.

    Returns:
        Moebius value of q given a of shape (n_instances,).

    """
    ps_a = powerset(a)  # powerset of A
    ps_a.pop(0)  # remove empty set
    m_a = np.zeros(q.shape[0])
    for b in ps_a:
        dl = len(set(a) - set(b))
        m_a += ((-1) ** dl) * capacity(q, b)
    return m_a
