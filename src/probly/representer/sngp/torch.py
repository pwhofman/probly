"""Torch specific functionality for the SNGP representer."""

from __future__ import annotations

from typing import Any

from probly.representation.distribution._common import create_categorical_distribution_from_logits
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
)
from probly.representation.sample.torch import TorchSample
from probly.representer.sngp._common import compute_categorical_sample_from_logits


@compute_categorical_sample_from_logits.register(TorchSample)
def torch_compute_categorical_sample_from_logits(
    sample: TorchSample[Any],
) -> TorchCategoricalDistributionSample:
    """Convert a TorchSample of SNGP logits to a categorical distribution sample."""
    tensor = sample.tensor
    sample_dim = sample.sample_dim
    if tensor.ndim >= 3 and sample_dim == 0:
        tensor = tensor.transpose(0, 1)
        sample_dim = 1

    categorical_dist = create_categorical_distribution_from_logits(tensor)
    return TorchCategoricalDistributionSample(tensor=categorical_dist, sample_dim=sample_dim)  # ty: ignore[invalid-argument-type]
