"""NumPy implementations of active learning query strategies."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from probly.quantification.measure.distribution.array import array_categorical_entropy
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution

from ._common import (
    badge_select,
    entropy_select,
    least_confident_select,
    margin_select,
    random_select,
    uncertainty_select,
)


@entropy_select.register(np.ndarray)
def _entropy_select_numpy(probs: np.ndarray, n: int) -> np.ndarray:
    """Numpy implementation of entropy-based selection."""
    h = array_categorical_entropy(ArrayCategoricalDistribution(probs))
    return np.argpartition(-h, n)[:n]


@least_confident_select.register(np.ndarray)
def _least_confident_select_numpy(probs: np.ndarray, n: int) -> np.ndarray:
    """Numpy implementation of least confident selection."""
    confidence = probs.max(axis=1)
    return np.argpartition(confidence, n)[:n]


@margin_select.register(np.ndarray)
def _margin_select_numpy(probs: np.ndarray, n: int) -> np.ndarray:
    """Numpy implementation of margin sampling selection."""
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argpartition(margin, n)[:n]


@uncertainty_select.register(np.ndarray)
def _uncertainty_select_numpy(scores: np.ndarray, n: int) -> np.ndarray:
    """Numpy implementation of uncertainty sampling selection."""
    return np.argpartition(scores, -n)[-n:]


@badge_select.register(np.ndarray)
def _badge_select_numpy(
    embeddings: np.ndarray,
    probs: np.ndarray,
    n: int,
    seed: int | None = None,
) -> np.ndarray:
    """Numpy implementation of BADGE selection."""
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
