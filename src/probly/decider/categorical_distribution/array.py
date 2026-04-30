"""Array-based deciders for reducing representations to categorical distributions."""

from __future__ import annotations

from probly.representation.conformal_set.array import ArrayOneHotConformalSet
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution

from ._common import categorical_from_mean


@categorical_from_mean.register(ArrayOneHotConformalSet)
def _(conformal_set: ArrayOneHotConformalSet) -> ArrayCategoricalDistribution:
    return ArrayCategoricalDistribution(conformal_set.array)
