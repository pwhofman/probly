"""Semantic entropy computation via NLI-based clustering.

Two variants following Kuhn et al., "Semantic Uncertainty" (Nature, 2024):
  - Discrete: cluster probabilities from sample counts (uniform weighting)
  - Weighted: cluster probabilities from generation log-likelihoods
"""

from __future__ import annotations

from collections import Counter
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gemma_experiment.entailment import EntailmentModel


def get_semantic_ids(responses: list[str], entailment_model: EntailmentModel) -> list[int]:
    """Cluster responses by semantic equivalence via bidirectional NLI.

    Two responses are semantically equivalent if neither direction is a
    contradiction (0) and they are not both neutral (1, 1).
    """
    semantic_ids = [-1] * len(responses)
    next_id = 0

    for i, text_i in enumerate(responses):
        if semantic_ids[i] != -1:
            continue
        semantic_ids[i] = next_id
        for j in range(i + 1, len(responses)):
            if semantic_ids[j] != -1:
                continue
            fwd = entailment_model.check_implication(text_i, responses[j])
            bwd = entailment_model.check_implication(responses[j], text_i)
            pair = [fwd, bwd]
            equivalent = (0 not in pair) and (pair != [1, 1])
            if equivalent:
                semantic_ids[j] = next_id
        next_id += 1

    return semantic_ids


def cluster_assignment_entropy(semantic_ids: list[int]) -> float:
    """Discrete semantic entropy: cluster probabilities from sample counts.

    Each generation is weighted equally. Cluster probability is simply
    the fraction of generations assigned to that cluster.

        p(C_k) = |{i : s_i in C_k}| / N
        SE_discrete = -sum_k p(C_k) * log(p(C_k))
    """
    n = len(semantic_ids)
    if n == 0:
        return 0.0
    counts = Counter(semantic_ids)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def weighted_semantic_entropy(
    semantic_ids: list[int],
    log_likelihoods: list[float],
) -> float:
    """Log-probability weighted semantic entropy (Kuhn et al., 2023).

    Cluster probabilities are derived from length-normalized generation
    log-likelihoods rather than uniform counts.

        log p(C_k) = logsumexp_{i in C_k}(ll_i) - logsumexp_j(ll_j)
        SE = -sum_k p(C_k) * log(p(C_k))

    Args:
        semantic_ids: Cluster assignment per generation.
        log_likelihoods: Length-normalized log-likelihood per generation.
    """
    if not semantic_ids:
        return 0.0

    ll = np.array(log_likelihoods)
    log_normalizer = _logsumexp(ll)

    unique_ids = sorted(set(semantic_ids))
    log_cluster_probs = []
    for uid in unique_ids:
        member_lls = ll[[i for i, sid in enumerate(semantic_ids) if sid == uid]]
        log_p = _logsumexp(member_lls) - log_normalizer
        log_cluster_probs.append(log_p)

    log_cluster_probs = np.array(log_cluster_probs)
    cluster_probs = np.exp(log_cluster_probs)
    return float(max(0.0, -np.sum(cluster_probs * log_cluster_probs)))


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = x.max()
    return float(c + np.log(np.sum(np.exp(x - c))))
