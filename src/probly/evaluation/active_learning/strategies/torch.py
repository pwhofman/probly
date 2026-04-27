"""PyTorch implementations of active learning query strategies."""

from __future__ import annotations

import numpy as np
import torch

from ._common import badge_select, margin_select, uncertainty_select


@margin_select.register(torch.Tensor)
def _margin_select_torch(probs: torch.Tensor, n: int) -> np.ndarray:
    sorted_probs = probs.sort(dim=1).values
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return torch.topk(margin, n, largest=False).indices.cpu().numpy()


@uncertainty_select.register(torch.Tensor)
def _uncertainty_select_torch(scores: torch.Tensor, n: int) -> np.ndarray:
    return torch.topk(scores, n, largest=True).indices.cpu().numpy()


@badge_select.register(torch.Tensor)
def _badge_select_torch(
    embeddings: torch.Tensor,
    probs: torch.Tensor,
    n: int,
    seed: int | None = None,
) -> np.ndarray:
    flat = embeddings.reshape(len(embeddings), -1)
    predicted_class = probs.argmax(dim=1)
    p_predicted = probs[torch.arange(len(probs)), predicted_class]
    grad_embeddings = flat * (1 - p_predicted).unsqueeze(1)

    rng = np.random.default_rng(seed)
    n_pool = len(grad_embeddings)

    first = int(rng.integers(0, n_pool))
    chosen: list[int] = [first]

    for _ in range(1, n):
        dists = torch.cdist(grad_embeddings, grad_embeddings[chosen]).pow(2)
        min_dists = dists.min(dim=1).values
        min_dists[chosen] = 0.0
        total = min_dists.sum()
        if total == 0.0:
            remaining = np.setdiff1d(np.arange(n_pool), chosen)
            if len(remaining) == 0:
                break
            next_idx = int(rng.choice(remaining))
        else:
            probs_sample = min_dists.cpu().numpy()
            probs_sample /= probs_sample.sum()
            next_idx = int(rng.choice(n_pool, p=probs_sample))
        chosen.append(next_idx)

    return np.array(chosen)
