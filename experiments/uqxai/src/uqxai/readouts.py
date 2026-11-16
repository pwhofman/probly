"""Readouts: the model outputs that are explained, as functions of the logits of the heads.

Every readout maps the logits of all heads, of shape ``(m, num_heads, num_classes)``, to the values
of one or more games, of shape ``(m, num_outputs)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import softmax

if TYPE_CHECKING:
    from collections.abc import Callable


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=-1)


def probability(logits: np.ndarray) -> np.ndarray:
    """Class probabilities of the Bayesian model average (the mean over the heads)."""
    return softmax(logits, axis=-1).mean(axis=1)


def log_probability(logits: np.ndarray) -> np.ndarray:
    """Logarithm of the class probabilities of the Bayesian model average."""
    return np.log(probability(logits))


def mean_logit(logits: np.ndarray) -> np.ndarray:
    """Class logits, averaged over the heads."""
    return logits.mean(axis=1)


def member_probability(logits: np.ndarray) -> np.ndarray:
    """Class probabilities of every head, flattened to ``num_heads * num_classes`` outputs."""
    return softmax(logits, axis=-1).reshape(len(logits), -1)


def uncertainty(logits: np.ndarray) -> np.ndarray:
    """Total, aleatoric and epistemic uncertainty of the entropy decomposition, in nats.

    Total uncertainty is the entropy of the Bayesian model average, aleatoric uncertainty the mean
    entropy of the heads, and epistemic uncertainty their difference (the mutual information).
    """
    probabilities = softmax(logits, axis=-1)
    total = _entropy(probabilities.mean(axis=1))
    aleatoric = _entropy(probabilities).mean(axis=1)
    return np.stack([total, aleatoric, total - aleatoric], axis=1)


READOUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "probability": probability,
    "log_probability": log_probability,
    "mean_logit": mean_logit,
    "member_probability": member_probability,
    "uncertainty": uncertainty,
}
UNCERTAINTY_NAMES = ("total", "aleatoric", "epistemic")
