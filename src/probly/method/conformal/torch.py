"""PyTorch-specific conformal prediction method implementations."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import ensure_distribution_2d


@ensure_distribution_2d.register(TorchSample)
def _(prediction: TorchSample) -> TorchCategoricalDistribution:
    """Ensure that the prediction is a distribution predictor.

    Aggregate over the sample dimension if the prediction has more than 2 dimensions.
    """
    prediction = prediction.move_sample_dim(0)
    data_t = prediction.tensor
    if data_t.ndim > 3:
        msg = "Predictions with more than 3 dimensions are not supported for conformal classification."
        raise ValueError(msg)
    if isinstance(data_t, TorchCategoricalDistribution):
        return data_t
    if data_t.ndim < 2:
        msg = "The predictor must return a distribution to be conformalized."
        raise ValueError(msg)
    if not torch.allclose(data_t.sum(dim=-1), torch.ones_like(data_t[..., 0])):
        data_t = torch.exp(data_t - torch.max(data_t, dim=-1, keepdim=True).values)
        data_t = data_t / data_t.sum(dim=-1, keepdim=True)
    if data_t.ndim == 3:
        data_t = data_t.mean(dim=0)
    return TorchCategoricalDistribution(unnormalized_probabilities=data_t)
