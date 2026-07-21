"""Ensemble-based uncertainty estimator using probly.

Wraps K independently-trained DQN agents. Uses probly's
ArrayCategoricalDistributionSample + quantify pipeline for proper
aleatoric/epistemic decomposition, and raw Q-value statistics for
action selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from probly.quantification import quantify
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)

from . import _softmax
from .interface import UncertaintyResult

if TYPE_CHECKING:
    from experiments.rl_uncertainty.agents.dqn import DQNAgent
    from probly.quantification.decomposition import AleatoricEpistemicTotalDecomposition


class EnsembleEstimator:
    """Uncertainty from an ensemble of DQN agents via probly.

    Args:
        agents: List of trained DQNAgent instances (one per ensemble member).
    """

    def __init__(self, agents: list[DQNAgent]) -> None:
        """Initialize with a list of trained DQN agents."""
        self._agents = agents

    def _stacked_q(self, states: np.ndarray) -> np.ndarray:
        """Stack Q-values from all members. Returns shape (K, batch, n_actions)."""
        return np.stack([a.batch_q_values(states) for a in self._agents], axis=0)

    def q_with_uncertainty(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of Q-values across ensemble. Each (batch, n_actions)."""
        stacked = self._stacked_q(states)  # (K, batch, n_actions)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def estimate(self, states: np.ndarray) -> UncertaintyResult:
        """Per-state uncertainty via probly's decomposition pipeline.

        Converts Q-values to softmax probabilities, then uses
        ArrayCategoricalDistributionSample + quantify to get
        aleatoric/epistemic/total decomposition per state.
        """
        stacked_q = self._stacked_q(states)  # (K, batch, n_actions)
        stacked_probs = _softmax(stacked_q)  # (K, batch, n_actions)

        batch_size = states.shape[0]
        epi = np.zeros(batch_size)
        alea = np.zeros(batch_size)
        total = np.zeros(batch_size)

        for i in range(batch_size):
            probs_i = stacked_probs[:, i, :]  # (K, n_actions)
            sample = ArrayCategoricalDistributionSample(
                array=ArrayCategoricalDistribution(unnormalized_probabilities=probs_i),
                sample_axis=0,
            )
            decomp = cast("AleatoricEpistemicTotalDecomposition", quantify(sample))
            epi[i] = float(decomp.epistemic)
            alea[i] = float(decomp.aleatoric)
            total[i] = float(decomp.total)

        return UncertaintyResult(epistemic=epi, aleatoric=alea, total=total)
