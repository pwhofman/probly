"""Shapley values of vector-valued games: one explanation per model output, on shared coalitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import shapiq

if TYPE_CHECKING:
    from collections.abc import Callable

APPROXIMATORS: dict[str, type[shapiq.Approximator]] = {
    "oddshap": shapiq.OddSHAP,
    "kernelshap": shapiq.KernelSHAP,
}


@dataclass(frozen=True)
class Attribution:
    """Shapley values of every output of a vector-valued game.

    Attributes:
        values: Shapley values of shape ``(n_players, n_outputs)``.
        empty: Value of the empty coalition per output, of shape ``(n_outputs,)``.
        full: Value of the grand coalition per output, of shape ``(n_outputs,)``.
    """

    values: np.ndarray
    empty: np.ndarray
    full: np.ndarray


def shapley_values(
    game: Callable[[np.ndarray], np.ndarray],
    n_players: int,
    budget: int,
    approximator: str = "oddshap",
    random_state: int = 0,
    **kwargs: Any,  # noqa: ANN401
) -> Attribution:
    """Approximate the Shapley values of every output of a vector-valued game.

    Each output is explained by its own approximator. All of them get the same ``random_state``
    and so sample the same coalitions: a ``game`` that caches its evaluations (see
    :class:`uqxai.game.CoalitionCache`) spends the budget once, not once per output.

    Args:
        game: Maps a boolean coalition matrix of shape ``(m, n_players)`` to values of shape
            ``(m, n_outputs)``.
        n_players: Number of players.
        budget: Number of coalitions each approximator may evaluate.
        approximator: ``"oddshap"`` or ``"kernelshap"``.
        random_state: Seed of the coalition sampling.
        **kwargs: Passed on to the approximator.

    Returns:
        The Shapley values of all outputs.
    """
    boundary = np.zeros((2, n_players), dtype=bool)
    boundary[1] = True
    empty, full = game(boundary)
    values = np.zeros((n_players, len(empty)))
    for output in range(len(empty)):
        estimator = APPROXIMATORS[approximator](n=n_players, random_state=random_state, **kwargs)
        estimate = estimator.approximate(budget, lambda coalitions, output=output: game(coalitions)[:, output])
        values[:, output] = estimate.get_n_order_values(1)
    return Attribution(values=values, empty=np.asarray(empty), full=np.asarray(full))
