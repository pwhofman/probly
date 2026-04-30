"""Smoke tests for DQN agent."""

from __future__ import annotations

import numpy as np


def test_dqn_select_action_shape():
    from experiments.rl_uncertainty.agents.dqn import DQNAgent

    agent = DQNAgent(state_dim=2, n_actions=4, seed=0)
    action = agent.select_action(np.array([0.5, 0.5]), epsilon=0.0)
    assert isinstance(action, int)
    assert 0 <= action < 4


def test_dqn_q_values_shape():
    from experiments.rl_uncertainty.agents.dqn import DQNAgent

    agent = DQNAgent(state_dim=2, n_actions=4, seed=0)
    q = agent.q_values(np.array([0.5, 0.5]))
    assert q.shape == (4,)


def test_dqn_batch_q_values():
    from experiments.rl_uncertainty.agents.dqn import DQNAgent

    agent = DQNAgent(state_dim=2, n_actions=4, seed=0)
    states = np.random.randn(10, 2).astype(np.float32)
    q = agent.batch_q_values(states)
    assert q.shape == (10, 4)


def test_dqn_learns_trivial():
    """DQN should learn to always pick action 0 when it gives +1 and others give -1."""
    from experiments.rl_uncertainty.agents.dqn import DQNAgent

    agent = DQNAgent(state_dim=1, n_actions=2, hidden=32, seed=42)
    state = np.array([0.0])
    for _ in range(500):
        agent.store(state, 0, 1.0, state, False)
        agent.store(state, 1, -1.0, state, False)
    for _ in range(200):
        agent.train_step(batch_size=32)
    q = agent.q_values(state)
    assert q[0] > q[1], f"Expected Q[0] > Q[1], got {q}"
