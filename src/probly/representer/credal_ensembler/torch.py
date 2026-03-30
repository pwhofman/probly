"""Torch specific functionality for credal ensembling representers."""

from __future__ import annotations

import torch

from probly.representer.credal_ensembler._common import compute_representative_set


@compute_representative_set.register(torch.Tensor)
def torch_compute_representative_set(probs: torch.Tensor, alpha: float, distance: str) -> torch.Tensor:
    """This function constructs a set of distributions based on :cite:`nguyenCredalEnsembling2025`.

    In general, a distribution is included in the set if it is in the top (1 - alpha) fraction of distributions closest
    to a representative distribution according to a specified distance metric.

    Args:
        probs: A tensor of shape (batch_size, num_members, num_classes) containing the
            predicted probabilities from the ensemble members.
        alpha: A float in the range [0, 1] that controls the size of the representative set.
            A value of 0 means all distributions are included, while a value of 1 means
            only the single closest distribution is included.
        distance: A string specifying the distance metric to use for determining closeness.

    """
    if distance == "euclidean":
        if alpha == 0.0:
            return probs
        mean = torch.mean(probs, dim=1)
        dist = torch.norm(probs - mean.unsqueeze(1), dim=-1)
        k = int(probs.shape[1] * (1 - alpha))
        k = max(k, 1)  # Ensure at least one member is selected
        _, idx = torch.topk(dist, k=k, dim=1, largest=False)
        selected_probs = torch.gather(probs, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, probs.shape[2]))
        return selected_probs
    msg = f"Distance {distance} not implemented for torch tensors."
    raise NotImplementedError(msg)
