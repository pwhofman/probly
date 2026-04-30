"""Credal relative likelihood uncertainty estimator.

Uses probly's credal_relative_likelihood to create a credal ensemble
from a single Q-network, then quantifies uncertainty via probly's pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from probly.method.credal_relative_likelihood import credal_relative_likelihood
from probly.predictor import predict_raw
from probly.quantification import quantify
from probly.quantification.decomposition import AleatoricEpistemicTotalDecomposition
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)

from .interface import UncertaintyResult

if TYPE_CHECKING:
    from experiments.rl_uncertainty.agents.dqn import DQNAgent


class CredalEstimator:
    """Uncertainty via credal relative likelihood.

    Creates a credal ensemble from a single DQN's Q-network using
    probly's credal_relative_likelihood, which biases each member
    toward a different action class.

    Args:
        agent: A trained DQNAgent whose Q-network will be wrapped.
        num_members: Number of credal ensemble members.
        tobias_value: Bias strength for credal initialization.
        seed: Random seed.
    """

    def __init__(
        self,
        agent: DQNAgent,
        num_members: int = 10,
        tobias_value: int = 100,
        seed: int = 0,
    ) -> None:
        self._agent = agent
        torch.manual_seed(seed)
        q_net = agent.get_network()
        self._credal_members = credal_relative_likelihood(
            q_net,
            num_members=num_members,
            reset_params=False,
            tobias_value=tobias_value,
        )

    def _stacked_q(self, states: np.ndarray) -> np.ndarray:
        """Q-values from each credal member. Returns (K, batch, n_actions)."""
        x = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            outputs = [predict_raw(m, x).numpy() for m in self._credal_members]
        return np.stack(outputs, axis=0)

    def q_with_uncertainty(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of Q-values across credal members. Each (batch, n_actions)."""
        stacked = self._stacked_q(states)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def estimate(self, states: np.ndarray) -> UncertaintyResult:
        """Per-state uncertainty via probly's decomposition pipeline.

        Converts Q-values to softmax probabilities per credal member, then uses
        ArrayCategoricalDistributionSample + quantify to get
        aleatoric/epistemic/total decomposition per state.
        """
        stacked_q = self._stacked_q(states)
        stacked_probs = _softmax(stacked_q)

        batch_size = states.shape[0]
        epi = np.zeros(batch_size)
        alea = np.zeros(batch_size)
        total = np.zeros(batch_size)

        for i in range(batch_size):
            probs_i = stacked_probs[:, i, :]
            sample = ArrayCategoricalDistributionSample(
                array=ArrayCategoricalDistribution(unnormalized_probabilities=probs_i),
                sample_axis=0,
            )
            decomp = cast("AleatoricEpistemicTotalDecomposition", quantify(sample))
            epi[i] = float(decomp.epistemic)
            alea[i] = float(decomp.aleatoric)
            total[i] = float(decomp.total)

        return UncertaintyResult(epistemic=epi, aleatoric=alea, total=total)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
