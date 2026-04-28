"""PyTorch implementations of active learning query strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from probly.quantification.measure.distribution.torch import torch_categorical_entropy
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution

from ._common import (
    badge_select,
    entropy_select,
    least_confident_select,
    margin_select,
    random_select,
    uncertainty_select,
)


@entropy_select.register(torch.Tensor)
def _entropy_select_torch(probs: torch.Tensor, n: int) -> torch.Tensor:
    """Torch implementation of entropy-based selection."""
    h = torch_categorical_entropy(TorchCategoricalDistribution(probs))
    return torch.topk(h, n, largest=True).indices


@least_confident_select.register(torch.Tensor)
def _least_confident_select_torch(probs: torch.Tensor, n: int) -> torch.Tensor:
    """Torch implementation of least confident selection."""
    confidence = probs.max(dim=1).values
    return torch.topk(confidence, n, largest=False).indices


@margin_select.register(torch.Tensor)
def _margin_select_torch(probs: torch.Tensor, n: int) -> torch.Tensor:
    """Torch implementation of margin sampling selection."""
    sorted_probs = probs.sort(dim=1).values
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return torch.topk(margin, n, largest=False).indices


@uncertainty_select.register(torch.Tensor)
def _uncertainty_select_torch(scores: torch.Tensor, n: int) -> torch.Tensor:
    """Torch implementation of uncertainty sampling selection."""
    return torch.topk(scores, n, largest=True).indices


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
