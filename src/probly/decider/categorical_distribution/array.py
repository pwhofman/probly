"""Array-based deciders for reducing representations to categorical distributions."""

from __future__ import annotations

import numpy as np

from probly.representation.conformal_set.array import ArrayOneHotConformalSet
from probly.representation.credal_set.array import ArrayConvexCredalSet
from probly.representation.distribution.array_categorical import ArrayProbabilityCategoricalDistribution

from ._common import categorical_from_maximin, categorical_from_mean


@categorical_from_mean.register(ArrayOneHotConformalSet)
def _(conformal_set: ArrayOneHotConformalSet) -> ArrayProbabilityCategoricalDistribution:
    return ArrayProbabilityCategoricalDistribution(conformal_set.array)


@categorical_from_maximin.register(ArrayConvexCredalSet)
def _(credal_set: ArrayConvexCredalSet) -> ArrayProbabilityCategoricalDistribution:
    lower = credal_set.lower()
    argmax = np.argmax(lower, axis=-1)
    one_hot = np.eye(credal_set.num_classes, dtype=lower.dtype)[argmax]
    return ArrayProbabilityCategoricalDistribution(one_hot)
