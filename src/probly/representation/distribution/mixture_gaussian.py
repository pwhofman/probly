"""Mixture distribution for Gaussian components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from probly.representation.sampling.array_sample import ArraySample

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from probly.representation.distribution.array_gaussian import ArrayGaussian


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

    def __array_namespace__(self) -> Any:  # noqa: ANN401
        """Get the array namespace of the underlying array."""
        return self.components[0].__array_namespace__()

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.components[0].dtype  # type: ignore[no-any-return]

    @property
    def device(self) -> str:
        """The device of the underlying array."""
        return self.components[0].device

    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return self.components[0].ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return self.components[0].shape

    @property
    def size(self) -> int:
        """The total number of elements in the underlying array."""
        return self.components[0].size

    @property
    def T(self) -> Self:  # noqa: N802
        """The transposed version of the mixture components."""
        return type(self)(
            components=[c.T for c in self.components],
            weights=self.weights,
        )

    def sample(
        self,
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> ArraySample:
        """Draw samples from the Gaussian mixture. Returns an ArraySample."""
        if rng is None:
            rng = np.random.default_rng()

        k = len(self.components)
        weights = self.weights

        comp_idx = rng.choice(k, size=num_samples, p=weights)

        reference_comp = self.components[0]
        reference_array = reference_comp.sample(1).array
        out_shape = (num_samples, *reference_array.shape[1:])
        result = np.empty(out_shape, dtype=reference_array.dtype)

        for k, component in enumerate(self.components):
            indices_for_component = comp_idx == k
            num_samples_for_component = int(indices_for_component.sum())
            if num_samples_for_component == 0:
                continue

            samples_for_component = component.sample(num_samples_for_component, rng=rng).array
            result[indices_for_component] = samples_for_component

        return ArraySample(array=result, sample_axis=0)
