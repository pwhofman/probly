"""Torch for SAPS."""

from __future__ import annotations

import torch

from .common import register


def saps_score_torch(
    probs: torch.Tensor,
    label: int,
    lambda_val: float = 0.1,
    u: float | None = None,
) -> float:
    """Compute SAPS Nonconformity Score for torch tensors.

    Args:
        probs: 1D tensor with softmax probabilities.
        label: true index.
        lambda_val: lambda value for SAPS.
        u: optional random value in [0,1).

    Returns:
        float: SAPS nonconformity score.
    """
    if probs.ndim == 2:
        if probs.shape[0] != 1:
            raise ValueError
        probs = probs[0]

    if probs.ndim != 1:
        raise ValueError

    if not (0 <= label < probs.shape[0]):
        raise ValueError

    if u is None:
        u = float(torch.rand(1).item())

    # get max probability
    max_probs = torch.max(probs).item()

    # get rank of label (1-based)
    sorted_indices = torch.argsort(probs, descending=True)
    rank_tensor = torch.where(sorted_indices == label)[0]

    if rank_tensor.numel() == 0:
        raise ValueError

    # convert to 1-based rank
    rank = int(rank_tensor[0].item()) + 1

    if rank == 1:
        return float(u * max_probs)
    return float(max_probs + (rank - 2 + u) * lambda_val)


# Optional batch helper function for Torch
def saps_score_torch_batch(
    probs: torch.Tensor,
    labels: torch.Tensor,
    lambda_val: float = 0.1,
    us: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch version of SAPS Nonconformity Score for torch tensors."""
    n_samples = probs.shape[0]

    if us is None:
        us = torch.rand(n_samples, device=probs.device)

    max_probs = torch.max(probs, dim=1).values

    sorted_indices = torch.argsort(probs, dim=1, descending=True)

    labels_expanded = labels.unsqueeze(1).expand(-1, probs.shape[1])
    rank_mask = sorted_indices == labels_expanded

    ranks = torch.argmax(rank_mask.float(), dim=1) + 1

    # Compute scores based on ranks
    scores = torch.where(
        ranks == 1,
        us * max_probs,
        max_probs + (ranks - 2 + us) * lambda_val,
    )

    return scores


register(torch.Tensor, saps_score_torch)
