"""Numpy-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, override

import numpy as np
from scipy import special

from probly.representation.distribution.common import DirichletDistribution

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayDirichletDistribution(
    DirichletDistribution,
    np.lib.mixins.NDArrayOperatorsMixin,
):
    """
    A Dirichlet distribution stored as a numpy array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    alphas: np.ndarray

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.alphas, np.ndarray):
            raise TypeError("alphas must be a numpy ndarray.")

        if self.alphas.ndim < 1:
            raise ValueError("alphas must have at least one dimension.")

        if np.any(self.alphas <= 0):
            raise ValueError("alphas must be strictly positive.")

        if self.alphas.shape[-1] < 2:
            raise ValueError("Dirichlet distribution requires at least 2 classes.")

    @classmethod
    def from_array(cls, alphas: np.ndarray | list, dtype: DTypeLike = None) -> Self:
        """Create a Dirichlet distribution from an array or list."""
        return cls(alphas=np.asarray(alphas, dtype=dtype))
