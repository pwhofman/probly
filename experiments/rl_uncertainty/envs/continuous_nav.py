"""Continuous 2D navigation environment.

Agent moves in a bounded 2D arena with 16 discrete actions (every 22.5 degrees).
Supports circular obstacles and rectangular walls.  Small Gaussian noise on
transitions provides a source of aleatoric uncertainty, amplified near obstacles
when turbulence is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import StepResult

# 16 evenly spaced directions (22.5 degree increments, starting from East=0)
_DIRS = np.array(
    [
        [np.cos(i * np.pi / 8), np.sin(i * np.pi / 8)]
        for i in range(16)
    ]
)

_DEFAULT_OBSTACLES: list[tuple[np.ndarray, float]] = [
    (np.array([0.5, 0.5]), 0.15),  # large center obstacle blocking direct path
    (np.array([0.3, 0.75]), 0.07),  # small upper-left obstacle
]

_GAUNTLET_OBSTACLES: list[tuple[np.ndarray, float]] = []

# Walls are axis-aligned rectangles: (x_min, y_min, x_max, y_max)
# Wall spanning the full width with ONE gap in the center.
# All agents must pass through — the only question is how aggressively.
_GAUNTLET_WALLS: list[tuple[float, float, float, float]] = [
    # Wall at y=0.50 spanning full width with ONE gap in the center.
    # Gap at x=[0.17, 0.33] (0.16 wide = ~3 steps). All agents must thread it.
    (0.00, 0.49, 0.17, 0.52),  # left segment
    (0.33, 0.49, 0.50, 0.52),  # right segment
]

_LAYOUTS: dict[str, dict] = {
    "default": {
        "obstacles": _DEFAULT_OBSTACLES,
        "start": np.array([0.1, 0.1]),
        "goal": np.array([0.9, 0.9]),
    },
    "gauntlet": {
        "obstacles": _GAUNTLET_OBSTACLES,
        "walls": _GAUNTLET_WALLS,
        "start": np.array([0.25, 0.1]),
        "goal": np.array([0.25, 0.9]),
        "bounds_low": np.array([0.0, 0.0]),
        "bounds_high": np.array([0.5, 1.0]),
        "turbulence": 0.02,
    },
}


def get_layout_obstacles(layout: str = "default") -> list[tuple[np.ndarray, float]]:
    """Return obstacles for a named layout."""
    return list(_LAYOUTS[layout]["obstacles"])


@dataclass
class ContinuousNavEnv:
    """Continuous 2D navigation with obstacles.

    Args:
        layout: Named layout preset ('default' or 'gauntlet').
        obstacles: List of (center, radius) tuples.
        start: Starting position.
        goal: Goal position.
        goal_radius: Distance to goal that counts as reaching it.
        step_size: How far the agent moves per action.
        noise_std: Gaussian noise on transitions (aleatoric uncertainty source).
        max_steps: Episode timeout.
    """

    layout: str = "default"
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
    turbulence: float = 0.0  # noise multiplier near obstacles (0=off)
    walls: list[tuple[float, float, float, float]] = field(default_factory=list)
    bounds_low: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    bounds_high: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))

    _pos: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _t: int = field(init=False, repr=False, default=0)
    _prev_dist: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        """Apply named layout configuration."""
        if self.layout != "default":
            cfg = _LAYOUTS[self.layout]
            self.obstacles = list(cfg.get("obstacles", []))
            self.walls = list(cfg.get("walls", []))
            self.start = cfg["start"].copy()
            self.goal = cfg["goal"].copy()
            for key in ("noise_std", "turbulence", "distance_shaping"):
                if key in cfg:
                    setattr(self, key, cfg[key])
            if "bounds_low" in cfg:
                self.bounds_low = cfg["bounds_low"].copy()
            if "bounds_high" in cfg:
                self.bounds_high = cfg["bounds_high"].copy()

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""
        return 2

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(_DIRS)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) bounds of the state space for heatmap gridding."""
        return self.bounds_low, self.bounds_high

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
            action: Integer in {0, ..., 15} for 16 compass directions
                (22.5 degree increments starting from East).

        Returns:
            StepResult with next state, reward, done flag, and info dict.
        """
        direction = _DIRS[action] * self.step_size

        # Turbulence: noise amplified near obstacles and walls
        effective_noise = self.noise_std
        if self.turbulence > 0:
            for center, radius in self.obstacles:
                dist = float(np.linalg.norm(self._pos - center))
                clearance = dist - radius
                if clearance < radius:
                    proximity = max(1.0 - clearance / radius, 0.0)
                    effective_noise += self.turbulence * proximity
            for xmin, ymin, xmax, ymax in self.walls:
                # Distance to nearest wall edge
                dx = max(xmin - self._pos[0], 0.0, self._pos[0] - xmax)
                dy = max(ymin - self._pos[1], 0.0, self._pos[1] - ymax)
                wall_dist = float(np.sqrt(dx * dx + dy * dy))
                zone = 0.10  # turbulence zone radius around walls
                if wall_dist < zone:
                    proximity = 1.0 - wall_dist / zone
                    effective_noise += self.turbulence * proximity

        noise = self._rng.normal(0, effective_noise, size=2)
        self._pos = np.clip(self._pos + direction + noise, self.bounds_low, self.bounds_high)
        self._t += 1

        cur_dist = float(np.linalg.norm(self._pos - self.goal))
        shaping = self.distance_shaping * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        # Check collision with circles
        for center, radius in self.obstacles:
            if np.linalg.norm(self._pos - center) < radius:
                return StepResult(self._pos.copy(), self.collision_reward, True, {"event": "collision"})

        # Check collision with walls (axis-aligned rectangles)
        for xmin, ymin, xmax, ymax in self.walls:
            if xmin <= self._pos[0] <= xmax and ymin <= self._pos[1] <= ymax:
                return StepResult(self._pos.copy(), self.collision_reward, True, {"event": "collision"})

        # Check goal
        if cur_dist < self.goal_radius:
            return StepResult(self._pos.copy(), self.goal_reward, True, {"event": "goal"})

        # Check timeout
        if self._t >= self.max_steps:
            return StepResult(self._pos.copy(), self.step_reward, True, {"event": "timeout"})

        return StepResult(self._pos.copy(), self.step_reward + shaping, False, {})
