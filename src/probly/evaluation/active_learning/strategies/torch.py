"""PyTorch implementations of active learning query strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from ._badge import badge_select
from ._scores import least_confident_score, margin_score
from ._selection import random_select, topk_select

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


@least_confident_score.register(torch.Tensor)
def _least_confident_score_torch(probs: torch.Tensor) -> torch.Tensor:
    """Torch implementation of least confident scoring."""
    return 1.0 - probs.max(dim=1).values


@margin_score.register(torch.Tensor)
def _margin_score_torch(probs: torch.Tensor) -> torch.Tensor:
    """Torch implementation of margin scoring (negative margin: higher = smaller margin)."""
    sorted_probs = probs.sort(dim=1).values
    return -(sorted_probs[:, -1] - sorted_probs[:, -2])


# ---------------------------------------------------------------------------
# Top-k selection
# ---------------------------------------------------------------------------


@topk_select.register(torch.Tensor)
def _topk_select_torch(scores: torch.Tensor, n: int) -> torch.Tensor:
    """Torch implementation of top-k selection (highest scores)."""
    return torch.topk(scores, n, largest=True).indices


# ---------------------------------------------------------------------------
# BADGE and Random
# ---------------------------------------------------------------------------


@badge_select.register(torch.Tensor)
def _badge_select_torch(
    embeddings: torch.Tensor,
    probs: torch.Tensor,
    n: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Torch implementation of BADGE selection."""
    flat = embeddings.reshape(len(embeddings), -1)
    predicted_class = probs.argmax(dim=1)
    p_predicted = probs[torch.arange(len(probs), device=probs.device), predicted_class]
    grad_embeddings = flat * (1 - p_predicted).unsqueeze(1)

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    if n <= 0:
        return torch.tensor([], dtype=torch.long, device=embeddings.device)

    n_pool = len(grad_embeddings)
    first = int(torch.randint(0, n_pool, (1,), generator=g))
    chosen: list[int] = [first]

    for _ in range(1, n):
        dists = torch.cdist(grad_embeddings, grad_embeddings[chosen]).pow(2)
        min_dists = dists.min(dim=1).values
        min_dists[chosen] = 0.0
        total = min_dists.sum()
        if total == 0.0:
            mask = torch.ones(n_pool, dtype=torch.bool, device=grad_embeddings.device)
            mask[chosen] = False
            remaining = mask.nonzero(as_tuple=False).squeeze(1)
            if len(remaining) == 0:
                break
            next_idx = int(remaining[int(torch.randint(len(remaining), (1,), generator=g))])
        else:
            next_idx = int(torch.multinomial(min_dists.cpu(), 1, generator=g))
        chosen.append(next_idx)

    return torch.tensor(chosen, dtype=torch.long, device=embeddings.device)


@random_select.register(torch.Tensor)
def _random_select_torch(
    x_ref: torch.Tensor,
    n_pool: int,
    n: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Torch implementation of random selection."""
    indices = rng.choice(n_pool, size=n, replace=False)
    return torch.from_numpy(indices).long().to(x_ref.device)
