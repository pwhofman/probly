"""Environment implementations for RL uncertainty experiments."""

from __future__ import annotations

from typing import Any

from .continuous_nav import ContinuousNavEnv
from .racetrack import RacetrackEnv


def make_env(name: str, **kwargs: Any) -> ContinuousNavEnv | RacetrackEnv:  # noqa: ANN401
    """Create an environment by name with optional overrides.

    Args:
        name: Either 'continuous_nav' or 'racetrack'.
        **kwargs: Passed to the environment constructor (e.g. collision_reward=-50).

    Returns:
        The instantiated environment.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name == "continuous_nav":
        return ContinuousNavEnv(**kwargs)
    if name == "racetrack":
        return RacetrackEnv(**kwargs)
    msg = f"Unknown env: {name}"
    raise ValueError(msg)
