"""Torch specific functionality for credal ensembling representers."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.representation.sample.torch import TorchSample
from probly.representer.credal_ensembler._common import (
    compute_credal_ensembling_set,
    compute_credal_net_set,
)


@compute_credal_ensembling_set.register(TorchSample)
def torch_compute_credal_ensembling_set(
    sample: TorchSample[TorchCategoricalDistribution],
    alpha: float,
    distance: str,
) -> TorchSample[TorchCategoricalDistribution]:
    """This function constructs a set of distributions based on :cite:`nguyenCredalEnsembling2025`.

    In general, a distribution is included in the set if it is in the top (1 - alpha) fraction of distributions closest
    to a representative distribution according to a specified distance metric.

    Args:
        sample: A sample containing the predicted probabilities from the ensemble members.
        alpha: A float in the range [0, 1] that controls the size of the representative set.
            A value of 0 means all distributions are included, while a value of 1 means
            only the single closest distribution is included.
        distance: A string specifying the distance metric to use for determining closeness.

    """
    if distance == "euclidean":
        if alpha == 0.0:
            return sample
        probs = sample.tensor.probabilities
        sample_dim = sample.sample_dim
        mean = torch.mean(probs, dim=sample_dim, keepdim=True)
        dist = torch.norm(probs - mean, dim=-1)
        k = int(sample.sample_size * (1 - alpha))
        k = max(k, 1)  # Ensure at least one member is selected
        _, idx = torch.topk(dist, k=k, dim=sample_dim, largest=False)
        selected_probs = torch.gather(probs, dim=sample_dim, index=idx.unsqueeze(-1).expand(-1, -1, probs.shape[-1]))
        return TorchCategoricalDistributionSample(
            tensor=TorchCategoricalDistribution(selected_probs),
            sample_dim=sample.sample_dim,
        )
    msg = f"Distance {distance} not implemented for torch tensors."
    raise NotImplementedError(msg)


@compute_credal_net_set.register(TorchSample)
def torch_compute_credal_net_set(
    sample: TorchSample[TorchCategoricalDistribution],
) -> TorchSample[TorchCategoricalDistribution]:
    """This function decouples the lower and upper bounds of the credal net."""
    probs = sample.tensor.probabilities
    probs = probs.reshape(*probs.shape[:-1], 2, probs.shape[-1] // 2)
    return TorchCategoricalDistributionSample(
        tensor=TorchCategoricalDistribution(probs),
        sample_dim=sample.sample_dim,
    )
