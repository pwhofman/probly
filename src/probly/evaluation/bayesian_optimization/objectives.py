"""Synthetic black-box objectives for Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from botorch.test_functions import Hartmann, Rosenbrock
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Objective:
    """A continuous black-box objective for BO over a bounded box.

    Attributes:
        name: Human-readable identifier used for logging.
        dim: Input dimensionality.
        bounds: Tensor of shape ``(2, dim)`` with row 0 = lower, row 1 = upper.
        optimal_value: Known global minimum value (used for regret).
        fn: Callable mapping ``(n, dim)`` inputs to ``(n,)`` outputs.
    """

    name: str
    dim: int
    bounds: torch.Tensor
    optimal_value: float
    fn: Callable[[torch.Tensor], torch.Tensor]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the objective at a batch of inputs."""
        return self.fn(x)


def rosenbrock(dim: int = 2) -> Objective:
    """Return the Rosenbrock objective in ``dim`` dimensions.

    Bounds are ``[-5, 10]`` along each axis (botorch default). The global
    minimum is 0 attained at ``(1, ..., 1)``.

    Args:
        dim: Input dimension. Must be at least 2.

    Returns:
        An :class:`Objective` wrapping ``botorch.test_functions.Rosenbrock``.
    """
    fn = Rosenbrock(dim=dim)
    return Objective(
        name=f"rosenbrock-{dim}d",
        dim=dim,
        bounds=cast("torch.Tensor", fn.bounds).clone().to(torch.float64),
        optimal_value=float(fn.optimal_value),
        fn=lambda x: fn.evaluate_true(x.to(torch.float64)),
    )


def hartmann() -> Objective:
    """Return the 6-D Hartmann objective with bounds ``[0, 1]^6``.

    The known global minimum is approximately ``-3.32237``.

    Returns:
        An :class:`Objective` wrapping ``botorch.test_functions.Hartmann``.
    """
    fn = Hartmann(dim=6)
    return Objective(
        name="hartmann-6d",
        dim=6,
        bounds=cast("torch.Tensor", fn.bounds).clone().to(torch.float64),
        optimal_value=float(fn.optimal_value),
        fn=lambda x: fn.evaluate_true(x.to(torch.float64)),
    )
