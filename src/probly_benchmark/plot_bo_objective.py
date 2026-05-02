"""Render a 1-D / 2-D BO objective plot, optionally overlaying a fitted surrogate.

Useful for sanity-checking surrogate posteriors visually after a short BO
run. Quick invocations::

    # Just the objective contour, no surrogate.
    uv run python -m probly_benchmark.plot_bo_objective --objective rosenbrock-2d

    # Forrester 1-D with a fitted GP surrogate after 8 Sobol observations.
    uv run python -m probly_benchmark.plot_bo_objective \
        --objective forrester-1d --surrogate gp --n-init 8 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from botorch.utils.sampling import draw_sobol_samples
import torch

from probly.evaluation.bayesian_optimization import (
    BNNSurrogate,
    BotorchGPSurrogate,
    MCDropoutSurrogate,
    Objective,
    RandomForestSurrogate,
    Surrogate,
    forrester,
    hartmann,
    plot_objective_1d,
    plot_objective_2d,
    rosenbrock,
)
from probly_benchmark.bayesian_optimization import DEFAULT_OUTPUT_DIR

if TYPE_CHECKING:
    from collections.abc import Callable

_OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "forrester-1d": forrester,
    "rosenbrock-2d": lambda: rosenbrock(dim=2),
    "hartmann-6d": hartmann,
}


def _make_surrogate(name: str, seed: int) -> Surrogate:
    """Construct a fresh, unfitted surrogate by short name."""
    match name:
        case "gp":
            return BotorchGPSurrogate()
        case "rf":
            return RandomForestSurrogate(n_estimators=200, seed=seed)
        case "dropout":
            return MCDropoutSurrogate(seed=seed)
        case "bnn":
            return BNNSurrogate(seed=seed)
        case _:
            msg = f"Unknown surrogate {name!r}. Available: 'gp', 'rf', 'dropout', 'bnn'."
            raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and render a BO objective plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        choices=sorted(_OBJECTIVES),
        default="rosenbrock-2d",
        help="Objective to plot. 'hartmann-6d' is included only for the 1-D scatter, since dim>2 has no contour.",
    )
    parser.add_argument(
        "--surrogate",
        choices=["gp", "rf", "dropout", "bnn", "none"],
        default="none",
        help="Optional fitted surrogate to overlay. Pass 'none' to plot the bare objective.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Sobol-sampled observations used to fit the surrogate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-objective",
        action="store_true",
        help="2-D only: contour log(f - f_opt) instead of raw f. Helpful for Rosenbrock.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used to resolve the default --output. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <output-dir>/<objective>__<surrogate>.png. Use --show to display.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively instead of saving.")
    args = parser.parse_args(argv)

    objective = _OBJECTIVES[args.objective]()
    if objective.dim not in (1, 2):
        msg = f"Visualization supports dim in (1, 2); got dim={objective.dim} for {objective.name}."
        raise ValueError(msg)

    surrogate = None
    x = y = None
    if args.surrogate != "none":
        torch.manual_seed(args.seed)
        x = draw_sobol_samples(bounds=objective.bounds, n=args.n_init, q=1, seed=args.seed).squeeze(-2)
        y = objective(x)
        surrogate = _make_surrogate(args.surrogate, args.seed)
        surrogate.fit(x, y)

    if objective.dim == 1:
        fig = plot_objective_1d(objective, surrogate=surrogate, x=x, y=y)
    else:
        fig = plot_objective_2d(objective, surrogate=surrogate, x=x, log_objective=args.log_objective)

    if args.show:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.show()
        return

    output: Path = (
        args.output if args.output is not None else args.output_dir / f"{objective.name}__{args.surrogate}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
