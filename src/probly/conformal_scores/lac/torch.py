"""LAC score computation for PyTorch tensors."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import lac_score


@lac_score.register(torch.Tensor)
def compute_lac_score_torch(probs: torch.Tensor, y_cal: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the LAC score."""
    probs_torch = torch.as_tensor(probs, dtype=torch.float)
    if probs_torch.ndim < 1:
        msg = (
            "probs must have at least one dimension with classes on the last axis, "
            f"got shape {tuple(probs_torch.shape)}."
        )
        raise ValueError(msg)

    scores = 1.0 - probs_torch
    if y_cal is not None:
        labels = torch.as_tensor(y_cal, device=probs_torch.device, dtype=torch.long)
        if tuple(labels.shape) != tuple(probs_torch.shape[:-1]):
            msg = (
                "y_cal must match probs batch shape (all axes except the class axis); "
                f"got y_cal shape {tuple(labels.shape)} and probs shape {tuple(probs_torch.shape)}."
            )
            raise ValueError(msg)
        scores = torch.gather(scores, -1, labels.unsqueeze(-1))
        scores = scores.squeeze(-1)
    return scores


@lac_score.register(TorchSample)
def compute_lac_score_torch_sample(probs: TorchSample, y_cal: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the LAC score for torch samples."""
    return lac_score(probs.tensor, y_cal)


@lac_score.register(TorchCategoricalDistribution)
def compute_lac_score_torch_categorical(
    probs: TorchCategoricalDistribution, y_cal: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute the LAC score for torch categorical distributions."""
    return compute_lac_score_torch(probs.probabilities, y_cal)
