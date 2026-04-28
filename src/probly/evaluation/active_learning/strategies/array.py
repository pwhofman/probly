"""NumPy implementations of active learning query strategies."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from probly.quantification.measure.distribution.array import array_categorical_entropy
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution

from ._badge import badge_select
from ._scores import entropy_score, least_confident_score, margin_score
from ._selection import random_select, topk_select

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


@entropy_score.register(np.ndarray)
def _entropy_score_numpy(probs: np.ndarray) -> np.ndarray:
    """Numpy implementation of entropy scoring."""
    return array_categorical_entropy(ArrayCategoricalDistribution(probs))


@least_confident_score.register(np.ndarray)
def _least_confident_score_numpy(probs: np.ndarray) -> np.ndarray:
    """Numpy implementation of least confident scoring."""
    return 1.0 - probs.max(axis=1)


@margin_score.register(np.ndarray)
def _margin_score_numpy(probs: np.ndarray) -> np.ndarray:
    """Numpy implementation of margin scoring (negative margin: higher = smaller margin)."""
    sorted_probs = np.sort(probs, axis=1)
    return -(sorted_probs[:, -1] - sorted_probs[:, -2])


# ---------------------------------------------------------------------------
# Top-k selection
# ---------------------------------------------------------------------------


@topk_select.register(np.ndarray)
def _topk_select_numpy(scores: np.ndarray, n: int) -> np.ndarray:
    """Numpy implementation of top-k selection (highest scores)."""
    if n >= len(scores):
        return np.arange(len(scores))
    return np.argpartition(-scores, n)[:n]


# ---------------------------------------------------------------------------
# BADGE and Random
# ---------------------------------------------------------------------------


@badge_select.register(np.ndarray)
def _badge_select_numpy(
    embeddings: np.ndarray,
    probs: np.ndarray,
    n: int,
    seed: int | None = None,
) -> np.ndarray:
    """Numpy implementation of BADGE selection."""
    if n <= 0:
        return np.array([], dtype=np.intp)
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


@random_select.register(np.ndarray)
def _random_select_numpy(
    x_ref: np.ndarray,  # noqa: ARG001
    n_pool: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Numpy implementation of random selection."""
    return rng.choice(n_pool, size=n, replace=False)
