"""2D racetrack environment.

Agent drives around an oval track with position + velocity state.
Actions: accelerate, brake, steer-left, steer-right, coast.
Track defined as corridor between inner and outer ellipses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import StepResult

# Actions: 0=accelerate, 1=brake, 2=steer-left, 3=steer-right, 4=coast
_ACCEL = 0.01
_BRAKE = -0.005
_STEER = 0.03
_FRICTION = 0.98
_MAX_SPEED = 0.05


@dataclass
class RacetrackEnv:
    """2D racetrack with oval corridor.

    Args:
        outer_a: Semi-axis of outer track boundary ellipse along x.
        outer_b: Semi-axis of outer track boundary ellipse along y.
        inner_a: Semi-axis of inner track boundary ellipse along x.
        inner_b: Semi-axis of inner track boundary ellipse along y.
        center: Center of the track ellipses.
        noise_std: Gaussian noise on acceleration (wind/friction).
        max_steps: Episode timeout.
        start_angle: Starting angle on the track (radians).
        finish_laps: Number of laps required to finish the episode.
    """

    outer_a: float = 0.4
    outer_b: float = 0.3
    inner_a: float = 0.2
    inner_b: float = 0.12
    center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    noise_std: float = 0.002
    max_steps: int = 300
    start_angle: float = 0.0
    finish_laps: int = 1

    _pos: np.ndarray = field(init=False, repr=False)
    _vel: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _t: int = field(init=False, repr=False, default=0)
    _angle_traveled: float = field(init=False, repr=False, default=0.0)
    _prev_angle: float = field(init=False, repr=False, default=0.0)

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""
        return 4

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return 5

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) bounds of the state space for heatmap gridding."""
        return (
            np.array([0.0, 0.0, -_MAX_SPEED, -_MAX_SPEED]),
            np.array([1.0, 1.0, _MAX_SPEED, _MAX_SPEED]),
        )

    def _on_track(self, pos: np.ndarray) -> bool:
        """Check if position is between inner and outer ellipses."""
        rel = pos - self.center
        outer_val = (rel[0] / self.outer_a) ** 2 + (rel[1] / self.outer_b) ** 2
        inner_val = (rel[0] / self.inner_a) ** 2 + (rel[1] / self.inner_b) ** 2
        return bool(inner_val >= 1.0 and outer_val <= 1.0)

    def _get_angle(self, pos: np.ndarray) -> float:
        rel = pos - self.center
        return float(np.arctan2(rel[1], rel[0]))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial state array of shape (4,) containing [x, y, vx, vy].
        """
        self._rng = np.random.default_rng(seed)
        mid_a = (self.outer_a + self.inner_a) / 2
        mid_b = (self.outer_b + self.inner_b) / 2
        self._pos = self.center + np.array([
            mid_a * np.cos(self.start_angle),
            mid_b * np.sin(self.start_angle),
        ])
        tangent = np.array([
            -np.sin(self.start_angle),
            np.cos(self.start_angle),
        ])
        self._vel = tangent * 0.01
        self._t = 0
        self._angle_traveled = 0.0
        self._prev_angle = self._get_angle(self._pos)
        return np.concatenate([self._pos, self._vel])

    def step(self, action: int) -> StepResult:
        """Take one environment step.

        Args:
            action: Integer action index. 0=accelerate, 1=brake,
                2=steer-left, 3=steer-right, 4=coast.

        Returns:
            StepResult with next_state, reward, done flag, and info dict.
            Reward is +1.0 per step on track, +20.0 on finish, -10.0 on
            wall collision, and 0.0 on timeout.
        """
        speed = np.linalg.norm(self._vel)
        if speed > 1e-8:
            forward = self._vel / speed
            leftward = np.array([-forward[1], forward[0]])
        else:
            forward = np.array([1.0, 0.0])
            leftward = np.array([0.0, 1.0])

        accel = np.zeros(2)
        if action == 0:  # accelerate
            accel = forward * _ACCEL
        elif action == 1:  # brake
            accel = forward * _BRAKE
        elif action == 2:  # steer left
            accel = leftward * _STEER
        elif action == 3:  # steer right
            accel = -leftward * _STEER
        # action == 4: coast

        noise = self._rng.normal(0, self.noise_std, size=2)
        self._vel = np.clip(
            (self._vel + accel + noise) * _FRICTION,
            -_MAX_SPEED, _MAX_SPEED,
        )
        self._pos = self._pos + self._vel
        self._t += 1

        state = np.concatenate([self._pos, self._vel])

        if not self._on_track(self._pos):
            return StepResult(state, -10.0, True, {"event": "wall"})

        cur_angle = self._get_angle(self._pos)
        delta = cur_angle - self._prev_angle
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi
        self._angle_traveled += delta
        self._prev_angle = cur_angle

        if self._angle_traveled >= 2 * np.pi * self.finish_laps:
            return StepResult(state, 20.0, True, {"event": "finish"})

        if self._t >= self.max_steps:
            return StepResult(state, 0.0, True, {"event": "timeout"})

        return StepResult(state, 1.0, False, {})
