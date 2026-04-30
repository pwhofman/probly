"""Penalized reward environment wrapper.

Shapes the reward during training by subtracting epistemic uncertainty:
    R' = R - lambda * EU(s)

This makes the agent learn to avoid high-uncertainty regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.rl_uncertainty.envs.base import Env, StepResult
from experiments.rl_uncertainty.uncertainty.interface import UncertaintyEstimator


@dataclass
class PenalizedRewardWrapper:
    """Wraps an environment to subtract epistemic uncertainty from rewards.

    Args:
        env: The base environment.
        estimator: Uncertainty estimator for EU computation.
        lambda_: Penalty strength. Higher = more risk-averse.
    """

    env: Env
    estimator: UncertaintyEstimator
    lambda_: float = 1.0

    @property
    def state_dim(self) -> int:
        return self.env.state_dim

    @property
    def n_actions(self) -> int:
        return self.env.n_actions

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.env.bounds

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self.env.reset(seed=seed)

    def step(self, action: int) -> StepResult:
        result = self.env.step(action)
        eu = float(self.estimator.estimate(result.next_state[np.newaxis]).epistemic[0])
        shaped_reward = result.reward - self.lambda_ * eu
        return StepResult(result.next_state, shaped_reward, result.done, result.info)
