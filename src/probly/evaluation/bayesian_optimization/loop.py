"""Bayesian optimization step iterator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from botorch.utils.sampling import draw_sobol_samples
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from probly.evaluation.bayesian_optimization.acquisition import Acquisition
    from probly.evaluation.bayesian_optimization.objectives import Objective
    from probly.evaluation.bayesian_optimization.surrogate import Surrogate


@dataclass
class BOState[S]:
    """State yielded after each BO iteration.

    Attributes:
        iteration: Current iteration number (0 = after the initial design).
        x: All observed inputs so far, shape ``(n_obs, dim)``.
        y: All observed outputs so far, shape ``(n_obs,)``.
        best_y: Minimum observed value so far.
        surrogate: The surrogate refit on all current observations.
    """

    iteration: int
    x: torch.Tensor
    y: torch.Tensor
    best_y: float
    surrogate: S


def bayesian_optimization_steps[S: Surrogate](
    objective: Objective,
    surrogate: S,
    acquisition: Acquisition,
    n_init: int = 10,
    n_iterations: int = 30,
    seed: int | None = None,
) -> Iterator[BOState[S]]:
    """Yield BO state after the initial design and each query-evaluate cycle.

    The loop:

    1. Draws ``n_init`` Sobol-sampled initial points and evaluates the
       objective at them.
    2. Fits the surrogate on the initial design and yields ``BOState``.
    3. Repeats: query the acquisition for the next input, evaluate the
       objective, refit the surrogate on the augmented dataset, yield.

    Args:
        objective: The black-box objective being minimized.
        surrogate: A surrogate implementing the
            :class:`~probly.evaluation.bayesian_optimization.surrogate.Surrogate`
            protocol.
        acquisition: An acquisition strategy implementing
            :class:`~probly.evaluation.bayesian_optimization.acquisition.Acquisition`.
        n_init: Number of initial Sobol-sampled evaluations.
        n_iterations: Number of acquisition rounds after the initial design.
        seed: Optional seed for the initial-design Sobol draw.

    Yields:
        ``BOState`` after the initial design (iteration=0) and after each
        subsequent query-refit cycle.
    """
    bounds = objective.bounds.to(torch.float64)
    x = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=seed).squeeze(-2)
    y = objective(x)
    surrogate.fit(x, y)
    yield BOState(iteration=0, x=x, y=y, best_y=float(y.min()), surrogate=surrogate)

    for i in range(n_iterations):
        x_new = acquisition.select(surrogate, objective.bounds)
        y_new = objective(x_new)
        x = torch.cat([x, x_new.to(x.dtype)], dim=0)
        y = torch.cat([y, y_new.to(y.dtype)], dim=0)
        surrogate.fit(x, y)
        yield BOState(iteration=i + 1, x=x, y=y, best_y=float(y.min()), surrogate=surrogate)
