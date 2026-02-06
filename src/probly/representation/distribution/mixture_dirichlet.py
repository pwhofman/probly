"""Mixture distribution for Dirichlet components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from probly.representation.sampling.sample import ArraySample

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from probly.representation.distribution.array_dirichlet import ArrayDirichlet


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayDirichletMixture:
    """Dirichlet mixture distribution."""

    components: Sequence[ArrayDirichlet]
    weights: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the mixture weights and component compatibility."""
        if len(self.components) == 0:
            msg = "components must not be empty."
            raise ValueError(msg)

        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 1:
            msg = "weights must be 1D -> (num_components,)."
            raise ValueError(msg)

        if w.shape[0] != len(self.components):
            msg = "weights must contain exactly one entry per component."
            raise ValueError(msg)

        if np.any(w < 0):
            msg = "weights must be non-negative."
            raise ValueError(msg)

        s = float(w.sum())
        if s <= 0:
            msg = "sum(weights) must be > 0."
            raise ValueError(msg)

        if not np.isclose(s, 1.0):
            w = w / s

        ref_shape = self.components[0].alphas.shape
        for i, c in enumerate(self.components):
            if c.alphas.shape != ref_shape:
                msg = (
                    f"component[{i}].alphas.shape={c.alphas.shape} does not match "
                    f"component[0].alphas.shape={ref_shape}."
                )
                raise ValueError(msg)

        object.__setattr__(self, "weights", w)

    def __array_namespace__(self) -> Any:  # noqa: ANN401
        """Get the array namespace of the underlying array."""
        # Keep consistent with ArrayDirichlet style.
        # If ArrayDirichlet returns self.alphas.__array_namespace__(), you may do the same.
        return self.components[0].__array_namespace__()

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.components[0].dtype  # type: ignore[no-any-return]

    @property
    def device(self) -> str:
        """The device of the underlying array."""
        return self.components[0].device  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        """Number of batch dimensions (excluding category axis)."""
        return self.components[0].ndim  # type: ignore[no-any-return]

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape (excluding category axis)."""
        return self.components[0].shape  # type: ignore[no-any-return]

    @property
    def size(self) -> int:
        """The total number of distributions."""
        return self.components[0].size

    @property
    def T(self) -> Self:  # noqa: N802
        """The transposed version of the distribution."""
        return type(self)(
            components=[c.T for c in self.components],
            weights=self.weights,
        )

    def num_sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> ArraySample:
        """Sample from the mixture.

        Returns:
        -------
        ArraySample
            Samples with shape: (num_samples, *alphas.shape).
        """
        if num_samples < 0:
            msg = "num_samples must be >= 0."
            raise ValueError(msg)

        if rng is None:
            rng = np.random.default_rng()

        k = len(self.components)
        comp_idx = rng.choice(k, size=num_samples, p=self.weights)

        ref_samples = self.components[0].sample(1, rng=rng).samples
        out = np.empty((num_samples, *ref_samples.shape[1:]), dtype=ref_samples.dtype)

        for j, comp in enumerate(self.components):
            mask = comp_idx == j
            n_j = int(mask.sum())
            if n_j == 0:
                continue

            s_j = comp.sample(n_j, rng=rng).samples
            out[mask] = s_j

        return ArraySample(array=out, sample_axis=0)
