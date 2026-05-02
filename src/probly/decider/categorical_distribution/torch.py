"""Torch implementations of deciders for categorical distributions."""

from __future__ import annotations

import torch

from probly.representation.conformal_set.torch import TorchOneHotConformalSet
from probly.representation.credal_set.torch import TorchConvexCredalSet
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution

from ._common import categorical_from_maximin, categorical_from_mean


@categorical_from_mean.register(TorchOneHotConformalSet)
def _(conformal_set: TorchOneHotConformalSet) -> TorchCategoricalDistribution:
    return TorchCategoricalDistribution(conformal_set.tensor)


@categorical_from_maximin.register(TorchConvexCredalSet)
def _(credal_set: TorchConvexCredalSet) -> TorchCategoricalDistribution:
    lower = credal_set.lower()
    argmax = torch.argmax(lower, dim=-1)
    one_hot = torch.nn.functional.one_hot(argmax, num_classes=credal_set.num_classes).to(dtype=lower.dtype)
    return TorchCategoricalDistribution(one_hot)
