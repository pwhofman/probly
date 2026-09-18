"""NumPy-based deciders for reducing representations to categorical distributions."""

from __future__ import annotations

import numpy as np

from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet
from probly.representation.credal_set.numpy import NumpyConvexCredalSet
from probly.representation.distribution.numpy_categorical import NumpyProbabilityCategoricalDistribution

from ._common import categorical_from_maximin, categorical_from_mean


@categorical_from_mean.register(NumpyOneHotConformalSet)
def _(conformal_set: NumpyOneHotConformalSet) -> NumpyProbabilityCategoricalDistribution:
    return NumpyProbabilityCategoricalDistribution(conformal_set.array)


@categorical_from_maximin.register(NumpyConvexCredalSet)
def _(credal_set: NumpyConvexCredalSet) -> NumpyProbabilityCategoricalDistribution:
    lower = credal_set.lower()
    argmax = np.argmax(lower, axis=-1)
    one_hot = np.eye(credal_set.num_classes, dtype=lower.dtype)[argmax]
    return NumpyProbabilityCategoricalDistribution(one_hot)
