"""Uncertainty estimator protocol.

Decouples RL agents from probly -- estimators consume agents and produce
uncertainty scalars. Swappable without changing anything in the RL or viz layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class UncertaintyResult:
    """Per-state uncertainty decomposition."""

    epistemic: np.ndarray  # shape (batch,)
    aleatoric: np.ndarray  # shape (batch,)
    total: np.ndarray  # shape (batch,)


class UncertaintyEstimator(Protocol):
    """Protocol for uncertainty estimators."""

    def estimate(self, states: np.ndarray) -> UncertaintyResult:
        """Uncertainty per state (for heatmap). States shape (batch, state_dim)."""
        ...

    def q_with_uncertainty(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (Q_mean, Q_std) each shaped (batch, n_actions)."""
        ...
