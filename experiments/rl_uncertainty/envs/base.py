"""Base environment protocol for RL uncertainty experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class StepResult:
    """Result of an environment step."""

    next_state: np.ndarray
    reward: float
    done: bool
    info: dict


class Env(Protocol):
    """Protocol for 2D RL environments."""

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset and return initial state."""
        ...

    def step(self, action: int) -> StepResult:
        """Take action, return (next_state, reward, done, info)."""
        ...

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""
        ...

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        ...

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) bounds of the state space for heatmap gridding."""
        ...
