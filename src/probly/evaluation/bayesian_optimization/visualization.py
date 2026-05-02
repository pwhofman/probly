"""Visualizations for Bayesian optimization on 1-D and 2-D objectives.

The plots are useful for sanity-checking surrogate posteriors against the
true objective on toy benchmarks (Forrester in 1-D, Rosenbrock in 2-D):

* :func:`plot_objective_1d` -- the 1-D objective curve, the surrogate's
  posterior mean ± std band, observed points, and an optional
  acquisition-score curve overlay.
* :func:`plot_objective_2d` -- the 2-D objective contour, observed points,
  and (optionally) the surrogate's mean and std contour panels.

Both helpers accept any :class:`~probly.evaluation.bayesian_optimization.surrogate.Surrogate`
because they read its posterior via ``predict(surrogate, x_grid)``; pass
``surrogate=None`` to draw only the objective and observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from probly.evaluation.bayesian_optimization.surrogate import posterior_mean_std
from probly.predictor import predict

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from probly.evaluation.bayesian_optimization.acquisition import Acquisition
    from probly.evaluation.bayesian_optimization.objectives import Objective
    from probly.evaluation.bayesian_optimization.surrogate import Surrogate


def _grid_1d(bounds: torch.Tensor, n: int) -> torch.Tensor:
    """Return a length-``n`` linspace inside ``bounds`` as an ``(n, 1)`` tensor."""
    lo, hi = float(bounds[0, 0]), float(bounds[1, 0])
    return torch.linspace(lo, hi, n, dtype=torch.float64).unsqueeze(-1)


def _grid_2d(bounds: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a ``(n, n)`` mesh inside ``bounds`` and the flattened ``(n*n, 2)`` query tensor."""
    lo = bounds[0].to(torch.float64)
    hi = bounds[1].to(torch.float64)
    xs = torch.linspace(float(lo[0]), float(hi[0]), n, dtype=torch.float64)
    ys = torch.linspace(float(lo[1]), float(hi[1]), n, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return grid_x, grid_y, flat


def plot_objective_1d(
    objective: Objective,
    *,
    surrogate: Surrogate | None = None,
    acquisition: Acquisition | None = None,
    x: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    n_grid: int = 400,
    beta: float = 2.0,
    ax: Axes | None = None,
) -> Figure:
    """Plot a 1-D objective with optional surrogate and acquisition overlays.

    Args:
        objective: 1-D objective to plot. Must have ``dim == 1``.
        surrogate: Optional fitted surrogate. If given, the posterior mean
            and ``± std`` band are drawn alongside the true objective.
        acquisition: Optional acquisition strategy. If given **and** the
            surrogate is provided, the acquisition score
            ``mean - beta * std`` (UCB convention for minimization) is
            drawn on a twin axis.
        x: Observed inputs of shape ``(n, 1)``. Drawn as scatter markers.
        y: Observed outputs of shape ``(n,)``.
        n_grid: Number of grid points used to evaluate the curves.
        beta: UCB beta used for the acquisition overlay. Ignored if no
            surrogate is given.
        ax: Optional matplotlib axes to draw into. A new figure is created
            if omitted.

    Returns:
        The matplotlib figure containing the plot.
    """
    if objective.dim != 1:
        msg = f"plot_objective_1d expects a 1-D objective, got dim={objective.dim}."
        raise ValueError(msg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = cast("Figure", ax.figure)

    grid = _grid_1d(objective.bounds, n_grid)
    grid_np = grid.squeeze(-1).numpy()

    with torch.no_grad():
        f_true = objective(grid).detach().cpu().numpy()
    ax.plot(grid_np, f_true, color="black", linewidth=1.6, label="objective")
    ax.axhline(objective.optimal_value, color="grey", linestyle=":", linewidth=1.0, label="optimum")

    if surrogate is not None:
        with torch.no_grad():
            mean_t, std_t = posterior_mean_std(predict(surrogate, grid))
        mean = mean_t.detach().cpu().numpy()
        std = std_t.detach().cpu().numpy()
        ax.plot(grid_np, mean, color="tab:blue", linewidth=1.4, label="surrogate mean")
        ax.fill_between(grid_np, mean - 2 * std, mean + 2 * std, color="tab:blue", alpha=0.18, label=r"$\pm 2\sigma$")

    if x is not None and y is not None:
        ax.scatter(
            x.detach().cpu().numpy().ravel(),
            y.detach().cpu().numpy().ravel(),
            color="tab:red",
            zorder=5,
            label="observations",
        )

    if surrogate is not None and acquisition is not None:
        with torch.no_grad():
            mean_t, std_t = posterior_mean_std(predict(surrogate, grid))
            acq = (mean_t - beta * std_t).detach().cpu().numpy()
        ax_twin = ax.twinx()
        ax_twin.plot(grid_np, acq, color="tab:green", linewidth=1.0, linestyle="--", label="acquisition")
        ax_twin.set_ylabel(r"acquisition ($\mu - \beta\sigma$)", color="tab:green")
        ax_twin.tick_params(axis="y", labelcolor="tab:green")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(objective.name)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    return fig


def plot_objective_2d(
    objective: Objective,
    *,
    surrogate: Surrogate | None = None,
    x: torch.Tensor | None = None,
    n_grid: int = 80,
    log_objective: bool = False,
) -> Figure:
    """Plot a 2-D objective contour with optional surrogate panels.

    With ``surrogate=None`` the figure has a single panel showing the true
    objective contour and any observations. With a fitted surrogate, two
    additional panels show the posterior mean and standard deviation over
    the same grid.

    Args:
        objective: 2-D objective to plot. Must have ``dim == 2``.
        surrogate: Optional fitted surrogate. If given, two extra panels
            (mean, std) are added to the figure.
        x: Observed inputs of shape ``(n, 2)``. Drawn as red scatter
            markers on every panel.
        n_grid: Resolution of the contour grid along each axis.
        log_objective: Use ``log(f - f_opt + eps)`` for the true-objective
            contour. Useful for objectives like Rosenbrock whose values
            span many orders of magnitude.

    Returns:
        The matplotlib figure containing the panels.
    """
    if objective.dim != 2:
        msg = f"plot_objective_2d expects a 2-D objective, got dim={objective.dim}."
        raise ValueError(msg)

    n_panels = 1 if surrogate is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.8), squeeze=False)
    panel_axes = list(axes[0])

    grid_x, grid_y, flat = _grid_2d(objective.bounds, n_grid)
    with torch.no_grad():
        z_true = objective(flat).reshape(n_grid, n_grid).detach().cpu().numpy()

    z_for_plot = np.log(z_true - float(objective.optimal_value) + 1e-9) if log_objective else z_true
    obj_label = "log(f - f_opt + eps)" if log_objective else "f(x)"

    cs = panel_axes[0].contourf(
        grid_x.numpy(),
        grid_y.numpy(),
        z_for_plot,
        levels=25,
        cmap="viridis",
    )
    fig.colorbar(cs, ax=panel_axes[0], shrink=0.85, label=obj_label)
    panel_axes[0].set_title(f"{objective.name}: objective")
    panel_axes[0].set_xlabel("x[0]")
    panel_axes[0].set_ylabel("x[1]")

    if surrogate is not None:
        with torch.no_grad():
            mean_t, std_t = posterior_mean_std(predict(surrogate, flat))
        mean = mean_t.reshape(n_grid, n_grid).detach().cpu().numpy()
        std = std_t.reshape(n_grid, n_grid).detach().cpu().numpy()

        cs_mean = panel_axes[1].contourf(grid_x.numpy(), grid_y.numpy(), mean, levels=25, cmap="viridis")
        fig.colorbar(cs_mean, ax=panel_axes[1], shrink=0.85, label="mean")
        panel_axes[1].set_title("surrogate mean")
        panel_axes[1].set_xlabel("x[0]")
        panel_axes[1].set_ylabel("x[1]")

        cs_std = panel_axes[2].contourf(grid_x.numpy(), grid_y.numpy(), std, levels=25, cmap="magma")
        fig.colorbar(cs_std, ax=panel_axes[2], shrink=0.85, label="std")
        panel_axes[2].set_title("surrogate std")
        panel_axes[2].set_xlabel("x[0]")
        panel_axes[2].set_ylabel("x[1]")

    if x is not None:
        x_np = x.detach().cpu().numpy()
        for panel in panel_axes:
            panel.scatter(x_np[:, 0], x_np[:, 1], color="red", s=22, edgecolor="white", linewidth=0.6, zorder=5)

    fig.tight_layout()
    return fig
