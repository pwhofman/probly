"""Tests for uncertainty estimators."""

from __future__ import annotations

import numpy as np


def test_ensemble_estimator_estimate():
    from experiments.rl_uncertainty.agents.dqn import DQNAgent
    from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator

    agents = [DQNAgent(state_dim=2, n_actions=4, seed=i) for i in range(5)]
    estimator = EnsembleEstimator(agents)

    states = np.random.randn(3, 2).astype(np.float32)
    result = estimator.estimate(states)
    assert result.epistemic.shape == (3,)
    assert result.aleatoric.shape == (3,)
    assert result.total.shape == (3,)
    assert np.all(result.epistemic >= 0)
    assert np.all(result.aleatoric >= 0)


def test_ensemble_estimator_q_with_uncertainty():
    from experiments.rl_uncertainty.agents.dqn import DQNAgent
    from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator

    agents = [DQNAgent(state_dim=2, n_actions=4, seed=i) for i in range(5)]
    estimator = EnsembleEstimator(agents)

    states = np.random.randn(3, 2).astype(np.float32)
    q_mean, q_std = estimator.q_with_uncertainty(states)
    assert q_mean.shape == (3, 4)
    assert q_std.shape == (3, 4)
    assert np.all(q_std >= 0)


def test_credal_estimator_estimate():
    from experiments.rl_uncertainty.agents.dqn import DQNAgent
    from experiments.rl_uncertainty.uncertainty.credal import CredalEstimator

    agent = DQNAgent(state_dim=2, n_actions=4, seed=0)
    estimator = CredalEstimator(agent, num_members=5, seed=0)

    states = np.random.randn(3, 2).astype(np.float32)
    result = estimator.estimate(states)
    assert result.epistemic.shape == (3,)
    assert result.aleatoric.shape == (3,)
    assert result.total.shape == (3,)


def test_credal_estimator_q_with_uncertainty():
    from experiments.rl_uncertainty.agents.dqn import DQNAgent
    from experiments.rl_uncertainty.uncertainty.credal import CredalEstimator

    agent = DQNAgent(state_dim=2, n_actions=4, seed=0)
    estimator = CredalEstimator(agent, num_members=5, seed=0)

    states = np.random.randn(3, 2).astype(np.float32)
    q_mean, q_std = estimator.q_with_uncertainty(states)
    assert q_mean.shape == (3, 4)
    assert q_std.shape == (3, 4)
