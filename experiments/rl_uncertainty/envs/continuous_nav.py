"""Continuous 2D navigation environment.

Agent moves in [0,1]^2 with discrete actions (up/down/left/right).
Circular obstacles end the episode on collision. Small Gaussian noise
on transitions provides a source of aleatoric uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import StepResult

# Actions: 0=up, 1=right, 2=down, 3=left
_DIRS = np.array([
    [0.0, 1.0],   # up
    [1.0, 0.0],   # right
    [0.0, -1.0],  # down
    [-1.0, 0.0],  # left
])

_DEFAULT_OBSTACLES: list[tuple[np.ndarray, float]] = [
    (np.array([0.5, 0.5]), 0.15),  # large center obstacle blocking direct path
    (np.array([0.3, 0.75]), 0.07),  # small upper-left obstacle
]


@dataclass
class ContinuousNavEnv:
    """Continuous 2D navigation with obstacles.

    Args:
        obstacles: List of (center, radius) tuples.
        start: Starting position.
        goal: Goal position.
        goal_radius: Distance to goal that counts as reaching it.
        step_size: How far the agent moves per action.
        noise_std: Gaussian noise on transitions (aleatoric uncertainty source).
        max_steps: Episode timeout.
    """

    obstacles: list[tuple[np.ndarray, float]] = field(default_factory=lambda: list(_DEFAULT_OBSTACLES))
    start: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1]))
    goal: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.9]))
    goal_radius: float = 0.08
    step_size: float = 0.05
    noise_std: float = 0.005
    max_steps: int = 200
    collision_reward: float = -25.0
    goal_reward: float = 50.0
    step_reward: float = 0.0
    distance_shaping: float = 20.0

    _pos: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _t: int = field(init=False, repr=False, default=0)
    _prev_dist: float = field(init=False, repr=False, default=0.0)

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""
        return 2

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return 4

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) bounds of the state space for heatmap gridding."""
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial state as a 2D position array.
        """
        self._rng = np.random.default_rng(seed)
        self._pos = self.start.copy()
        self._t = 0
        self._prev_dist = float(np.linalg.norm(self._pos - self.goal))
        return self._pos.copy()

    def step(self, action: int) -> StepResult:
        """Take a discrete action and advance the environment.

        Args:
            action: Integer in {0, 1, 2, 3} for up/right/down/left.

        Returns:
            StepResult with next state, reward, done flag, and info dict.
        """
        direction = _DIRS[action] * self.step_size
        noise = self._rng.normal(0, self.noise_std, size=2)
        self._pos = np.clip(self._pos + direction + noise, 0.0, 1.0)
        self._t += 1

        cur_dist = float(np.linalg.norm(self._pos - self.goal))
        shaping = self.distance_shaping * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        # Check collision
        for center, radius in self.obstacles:
            if np.linalg.norm(self._pos - center) < radius:
                return StepResult(self._pos.copy(), self.collision_reward, True, {"event": "collision"})

        # Check goal
        if cur_dist < self.goal_radius:
            return StepResult(self._pos.copy(), self.goal_reward, True, {"event": "goal"})

        # Check timeout
        if self._t >= self.max_steps:
            return StepResult(self._pos.copy(), self.step_reward, True, {"event": "timeout"})

        return StepResult(self._pos.copy(), self.step_reward + shaping, False, {})
