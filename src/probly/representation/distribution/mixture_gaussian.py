"""Mixture distribution for Gaussian components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from probly.representation.sampling.array_sample import ArraySample

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from probly.representation.distribution.array_gaussian import ArrayGaussian

_RNG = np.random.default_rng()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayGaussianMixture:
    """Gaussian mixture."""

    components: Sequence[ArrayGaussian]
    weights: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the mixture weights."""
        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 1:
            msg = "weights must be 1D -> (K,)."
            raise ValueError(msg)

        if len(self.components) != w.shape[0]:
            msg = "for every components there must be just one  weights."
            raise ValueError(msg)

        if np.any(w < 0):
            msg = "weights must be non-negative."
            raise ValueError(msg)

        s = w.sum()

        if not np.isclose(s, 1.0):
            w = w / s

        object.__setattr__(self, "weights", w)

    def sample(self, size: int) -> ArraySample:
        """Draw samples from the Gaussian mixture. Returns an ArraySample."""
        k = len(self.components)
        weights = self.weights

        comp_idx = _RNG.choice(k, size=size, p=weights)

        reference_comp = self.components[0]
        reference_array = reference_comp.sample(1).array
        out_shape = (size, *reference_array.shape[1:])
        result = np.empty(out_shape, dtype=reference_array.dtype)

        for k, component in enumerate(self.components):
            indices_for_component = comp_idx == k
            num_samples_for_component = int(indices_for_component.sum())
            if num_samples_for_component == 0:
                continue

            samples_for_component = component.sample(num_samples_for_component).array
            result[indices_for_component] = samples_for_component

        return ArraySample(array=result, sample_axis=0)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying component array."""
        return self.components.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the component array."""
        return self.components.ndim

    @property
    def size(self) -> int:
        """Return the total number of elements in the component array."""
        return self.components.size

    @property
    def dtype(self) -> DTypeLike:
        """Return the data type of the underlying component array."""
        return self.components.dtype
