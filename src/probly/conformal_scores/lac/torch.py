"""LAC score computation for PyTorch tensors."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import lac_score_func


@lac_score_func.register(torch.Tensor)
def compute_lac_score_torch(probs: torch.Tensor, y_cal: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the LAC score."""
    scores = 1.0 - probs
    if y_cal is not None:
        scores = scores[torch.arange(len(probs)), y_cal]
    return scores


@lac_score_func.register(TorchSample)
def compute_lac_score_torch_sample(probs: TorchSample, y_cal: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the LAC score for torch samples."""
    return compute_lac_score_torch(probs.tensor, y_cal)


@lac_score_func.register(TorchCategoricalDistribution)
def compute_lac_score_torch_categorical(
    probs: TorchCategoricalDistribution, y_cal: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute the LAC score for torch categorical distributions."""
    return compute_lac_score_torch(probs.probabilities, y_cal)
