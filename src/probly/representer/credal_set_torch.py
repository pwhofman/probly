"""Torch functionality for generic credal-set representers."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.representation.sample.torch import TorchSample
from probly.representer.credal_set import compute_representative_sample


@compute_representative_sample.register(TorchSample)
def torch_compute_representative_sample(
    sample: TorchSample[TorchCategoricalDistribution],
    alpha: float,
    distance: str,
) -> TorchSample[TorchCategoricalDistribution]:
    """Select predictions closest to the sample mean under the requested distance."""
    if distance == "euclidean":
        if alpha == 0.0:
            return sample
        probs = sample.tensor.probabilities
        sample_dim = sample.sample_dim
        mean = torch.mean(probs, dim=sample_dim, keepdim=True)
        dist = torch.norm(probs - mean, dim=-1)
        k = int(sample.sample_size * (1 - alpha))
        k = max(k, 1)
        _, idx = torch.topk(dist, k=k, dim=sample_dim, largest=False)
        selected_probs = torch.gather(probs, dim=sample_dim, index=idx.unsqueeze(-1).expand(-1, -1, probs.shape[-1]))
        return TorchCategoricalDistributionSample(
            tensor=TorchCategoricalDistribution(selected_probs),
            sample_dim=sample.sample_dim,
        )
    msg = f"Distance {distance} not implemented for torch tensors."
    raise NotImplementedError(msg)
