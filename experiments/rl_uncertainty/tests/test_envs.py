"""Smoke tests for RL environments."""

from __future__ import annotations

import numpy as np


def test_continuous_nav_reset_shape():
    from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv

    env = ContinuousNavEnv()
    state = env.reset(seed=0)
    assert state.shape == (2,)
    assert env.state_dim == 2
    assert env.n_actions == 4


def test_continuous_nav_step():
    from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv

    env = ContinuousNavEnv()
    env.reset(seed=0)
    result = env.step(0)  # up
    assert result.next_state.shape == (2,)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_continuous_nav_reaches_goal():
    """Agent moving toward goal should eventually get +10 reward."""
    from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv

    env = ContinuousNavEnv(
        obstacles=[],
        start=np.array([0.0, 0.0]),
        goal=np.array([0.15, 0.0]),
        goal_radius=0.06,
        noise_std=0.0,
    )
    env.reset(seed=0)
    for _ in range(3):
        result = env.step(1)  # right
    assert result.done
    assert result.reward == 10.0


def test_continuous_nav_collision():
    """Walking into an obstacle gives -10 and ends episode."""
    from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv

    env = ContinuousNavEnv(
        obstacles=[(np.array([0.1, 0.5]), 0.06)],
        start=np.array([0.05, 0.5]),
        goal=np.array([0.9, 0.9]),
        noise_std=0.0,
    )
    env.reset(seed=0)
    result = env.step(1)  # right -> into obstacle
    assert result.done
    assert result.reward == -10.0


def test_continuous_nav_bounds():
    from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv

    env = ContinuousNavEnv()
    low, high = env.bounds
    assert np.array_equal(low, np.array([0.0, 0.0]))
    assert np.array_equal(high, np.array([1.0, 1.0]))
