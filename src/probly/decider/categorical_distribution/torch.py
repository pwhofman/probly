"""Torch implementations of deciders for categorical distributions."""

from __future__ import annotations

from probly.representation.conformal_set.torch import TorchOneHotConformalSet
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution

from ._common import categorical_from_mean


@categorical_from_mean.register(TorchOneHotConformalSet)
def _(conformal_set: TorchOneHotConformalSet) -> TorchCategoricalDistribution:
    return TorchCategoricalDistribution(conformal_set.tensor)
