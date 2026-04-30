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
    (np.array([0.3, 0.6]), 0.08),
    (np.array([0.5, 0.3]), 0.10),
    (np.array([0.7, 0.7]), 0.07),
    (np.array([0.6, 0.5]), 0.06),
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
    goal_radius: float = 0.05
    step_size: float = 0.05
    noise_std: float = 0.005
    max_steps: int = 200

    _pos: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _t: int = field(init=False, repr=False, default=0)

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

        # Check collision
        for center, radius in self.obstacles:
            if np.linalg.norm(self._pos - center) < radius:
                return StepResult(self._pos.copy(), -10.0, True, {"event": "collision"})

        # Check goal
        if np.linalg.norm(self._pos - self.goal) < self.goal_radius:
            return StepResult(self._pos.copy(), 10.0, True, {"event": "goal"})

        # Check timeout
        if self._t >= self.max_steps:
            return StepResult(self._pos.copy(), -1.0, True, {"event": "timeout"})

        return StepResult(self._pos.copy(), -1.0, False, {})
