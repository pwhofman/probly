"""NumPy implementations of active learning query strategies."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._common import badge_select, margin_select, uncertainty_select


@margin_select.register(np.ndarray)
def _margin_select_numpy(probs: np.ndarray, n: int) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argpartition(margin, n)[:n]


@uncertainty_select.register(np.ndarray)
def _uncertainty_select_numpy(scores: np.ndarray, n: int) -> np.ndarray:
    return np.argpartition(scores, -n)[-n:]


@badge_select.register(np.ndarray)
def _badge_select_numpy(
    embeddings: np.ndarray,
    probs: np.ndarray,
    n: int,
    seed: int | None = None,
) -> np.ndarray:
    flat = embeddings.reshape(len(embeddings), -1)
    predicted_class = probs.argmax(axis=1)
    p_predicted = probs[np.arange(len(probs)), predicted_class]
    grad_embeddings = flat * (1 - p_predicted)[:, np.newaxis]

    rng = np.random.default_rng(seed)
    n_pool = len(grad_embeddings)

    first = int(rng.integers(0, n_pool))
    chosen: list[int] = [first]

    for _ in range(1, n):
        dists = cdist(grad_embeddings, grad_embeddings[chosen], metric="sqeuclidean")
        min_dists = dists.min(axis=1)
        min_dists[chosen] = 0.0
        total = min_dists.sum()
        if total == 0.0:
            remaining = np.setdiff1d(np.arange(n_pool), chosen)
            if len(remaining) == 0:
                break
            next_idx = int(rng.choice(remaining))
        else:
            probs_sample = min_dists / min_dists.sum()
            next_idx = int(rng.choice(n_pool, p=probs_sample))
        chosen.append(next_idx)

    return np.array(chosen)
