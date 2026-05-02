r"""Run BO on Rosenbrock and Hartmann6 with GP, random-forest, MC-Dropout, and BNN surrogates.

Compares :class:`BotorchGPSurrogate`, :class:`RandomForestSurrogate`,
:class:`MCDropoutSurrogate`, and :class:`BNNSurrogate` under
:class:`UpperConfidenceBound` acquisition. All four expose the same
probly representation-predictor interface (``predict(surrogate, x)``
returns a Gaussian distribution); the NN-based ones go through probly's
canonical UQ stack (``dropout()`` / ``bayesian()`` transformation +
``representer(num_samples=N)`` Sampler), the same wiring used by the
active-learning benchmark for these methods.

By default the per-objective budget (initial Sobol design + acquisition
rounds) is read from :func:`default_budget` -- a dimension-dependent table
calibrated for synthetic BO benchmarks. Pass ``--n-init`` and
``--n-iterations`` to override.

Run the default grid (writes ``<project>/bo_outputs/bo_results.json``)::

    uv run python -m probly_benchmark.bayesian_optimization

Override seeds, budget, and output location::

    uv run python -m probly_benchmark.bayesian_optimization \
        --n-init 10 --n-iterations 30 --seeds 0 1 2 \
        --surrogates gp rf --output-dir runs/bo-2026-05
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import fcntl
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from probly.evaluation.bayesian_optimization import (
    BNNSurrogate,
    BotorchGPSurrogate,
    MCDropoutSurrogate,
    Objective,
    RandomForestSurrogate,
    Surrogate,
    UpperConfidenceBound,
    bayesian_optimization_steps,
    hartmann,
    regret_nauc,
    rosenbrock,
    simple_regret,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "bo_outputs"
"""Default directory for BO benchmark outputs (``<project>/bo_outputs``)."""


_N_INIT_BY_DIM: dict[int, int] = {
    1: 12,
    2: 15,
    3: 18,
    4: 20,
    5: 22,
    6: 23,
    7: 25,
    8: 26,
    9: 28,
    10: 29,
    11: 30,
    12: 31,
    13: 33,
    14: 34,
    15: 35,
    16: 36,
    17: 37,
    18: 38,
    19: 39,
    20: 39,
    32: 49,
    50: 60,
    60: 66,
    100: 84,
    180: 111,
}

_N_ITERS_BY_DIM: dict[int, int] = {
    1: 60,
    2: 77,
    3: 90,
    4: 100,
    5: 110,
    6: 118,
    7: 126,
    8: 134,
    9: 140,
    10: 147,
    11: 153,
    12: 159,
    13: 165,
    14: 170,
    15: 175,
    16: 180,
    17: 185,
    18: 190,
    19: 195,
    20: 199,
    32: 247,
    50: 303,
    60: 330,
    100: 420,
    180: 330,
}


def default_budget(dim: int) -> tuple[int, int]:
    """Return ``(n_init, n_iterations)`` calibrated for a ``dim``-dimensional problem.

    The lookup follows a synthetic-BO benchmark convention: roughly
    ``n_init`` grows like ``O(d)`` and ``n_iterations`` like ``O(d log d)``.

    Args:
        dim: Input dimensionality of the objective.

    Returns:
        Tuple ``(n_init, n_iterations)`` sized for ``dim``.

    Raises:
        KeyError: If ``dim`` has no entry in the lookup. Pass
            ``--n-init``/``--n-iterations`` explicitly in that case.
    """
    if dim not in _N_INIT_BY_DIM or dim not in _N_ITERS_BY_DIM:
        msg = f"No default budget defined for dim={dim}. Pass --n-init and --n-iterations explicitly."
        raise KeyError(msg)
    return _N_INIT_BY_DIM[dim], _N_ITERS_BY_DIM[dim]


@dataclass
class IterationRecord:
    """Per-iteration log entry for a single BO run."""

    iteration: int
    n_obs: int
    best_y: float
    regret: float


@dataclass
class RunResult:
    """Summary and trajectory of a single BO run."""

    objective: str
    surrogate: str
    seed: int
    optimal_value: float
    n_init: int
    n_iterations: int
    final_best_y: float
    final_regret: float
    nauc: float
    iterations: list[IterationRecord] = field(default_factory=list)


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


def _run_one(
    objective: Objective,
    surrogate_name: str,
    *,
    n_init: int,
    n_iterations: int,
    beta: float,
    seed: int,
) -> RunResult:
    """Run BO once and return a summary including the per-iteration trajectory."""
    torch.manual_seed(seed)
    surrogate = _make_surrogate(surrogate_name, seed)
    acquisition = UpperConfidenceBound(beta=beta, seed=seed)

    iterations: list[IterationRecord] = []
    last_y: torch.Tensor = torch.empty(0)
    last_best: float = float("inf")
    running_best: float = float("inf")
    for state in bayesian_optimization_steps(
        objective,
        surrogate,
        acquisition,
        n_init=n_init,
        n_iterations=n_iterations,
        seed=seed,
    ):
        last_y = state.y
        last_best = state.best_y

        if state.iteration == 0:
            # Expand the initial Sobol design into per-evaluation records so
            # the budget curve covers evals 1..n_init, not just n_init.
            for i, val in enumerate(state.y.detach().cpu().tolist(), start=1):
                running_best = min(running_best, float(val))
                iterations.append(
                    IterationRecord(
                        iteration=0,
                        n_obs=i,
                        best_y=running_best,
                        regret=simple_regret(running_best, objective.optimal_value),
                    )
                )
        else:
            running_best = min(running_best, float(state.best_y))
            iterations.append(
                IterationRecord(
                    iteration=state.iteration,
                    n_obs=int(state.y.numel()),
                    best_y=running_best,
                    regret=simple_regret(running_best, objective.optimal_value),
                )
            )

        logger.info(
            "obj=%s surr=%s seed=%d iter=%d n_obs=%d best=%.4f regret=%.4f",
            objective.name,
            surrogate_name,
            seed,
            state.iteration,
            state.y.numel(),
            running_best,
            simple_regret(running_best, objective.optimal_value),
        )

    nauc = regret_nauc(last_y, objective.optimal_value)
    return RunResult(
        objective=objective.name,
        surrogate=surrogate_name,
        seed=seed,
        optimal_value=objective.optimal_value,
        n_init=n_init,
        n_iterations=n_iterations,
        final_best_y=last_best,
        final_regret=simple_regret(last_best, objective.optimal_value),
        nauc=nauc,
        iterations=iterations,
    )


def run_grid(
    objectives: Iterable[Objective],
    surrogate_names: Iterable[str],
    seeds: Iterable[int],
    *,
    budget_for: Callable[[Objective], tuple[int, int]] | None = None,
    beta: float = 2.0,
    checkpoint_file: Path | None = None,
) -> list[RunResult]:
    """Run BO over the cartesian product of objectives, surrogates, and seeds.

    Args:
        objectives: Iterable of :class:`Objective` instances to evaluate.
        surrogate_names: Iterable of surrogate identifiers
            (currently ``"gp"`` and ``"ensemble"``).
        seeds: Iterable of integer seeds; one BO run per seed.
        budget_for: Callable mapping an :class:`Objective` to
            ``(n_init, n_iterations)``. Defaults to :func:`default_budget`
            on the objective's dimensionality.
        beta: UCB exploration coefficient.
        checkpoint_file: If given, append each run's result to this JSON
            file as it completes. Lets a long-running grid be resumed or
            inspected partway through.

    Returns:
        A list of :class:`RunResult` -- one per ``(objective, surrogate, seed)``.
    """

    def _resolve_budget(obj: Objective) -> tuple[int, int]:
        return default_budget(obj.dim)

    resolve = budget_for if budget_for is not None else _resolve_budget
    results: list[RunResult] = []
    for objective in objectives:
        n_init, n_iterations = resolve(objective)
        for surrogate_name in surrogate_names:
            for seed in seeds:
                result = _run_one(
                    objective,
                    surrogate_name,
                    n_init=n_init,
                    n_iterations=n_iterations,
                    beta=beta,
                    seed=seed,
                )
                results.append(result)
                if checkpoint_file is not None:
                    _append_results(checkpoint_file, [result])
    return results


def _print_table(results: list[RunResult]) -> None:
    """Print a flat per-run summary."""
    print(f"{'objective':<14} {'surrogate':<10} {'seed':>4} {'best':>10} {'regret':>10} {'nauc':>8}")
    for r in results:
        print(
            f"{r.objective:<14} {r.surrogate:<10} {r.seed:>4} {r.final_best_y:>10.4f} "
            f"{r.final_regret:>10.4f} {r.nauc:>8.4f}"
        )


def _append_results(results_file: Path, results: list[RunResult]) -> None:
    """Append run summaries to a shared JSON list, deduplicating by ``(obj, surr, seed)``.

    Uses an exclusive ``fcntl`` lock so concurrent writers cannot corrupt
    the file -- mirrors the active-learning benchmark convention.

    Args:
        results_file: Destination JSON file. Created if missing.
        results: Run summaries to merge into the file.
    """
    results_file.parent.mkdir(parents=True, exist_ok=True)
    new_payload: list[dict[str, Any]] = [asdict(r) for r in results]
    new_keys = {(r["objective"], r["surrogate"], r["seed"]) for r in new_payload}

    with results_file.open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            existing: list[dict[str, Any]] = json.loads(content) if content.strip() else []
            existing = [r for r in existing if (r["objective"], r["surrogate"], r["seed"]) not in new_keys]
            existing.extend(new_payload)
            f.seek(0)
            f.truncate()
            json.dump(existing, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args, run the grid of BO experiments, and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rosenbrock-dim", type=int, default=2)
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Override the dim-dependent initial design size.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=None,
        help="Override the dim-dependent number of acquisition rounds.",
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--surrogates", nargs="+", default=["gp", "rf", "dropout", "bnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for default outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="JSON file for run summaries. Defaults to <output-dir>/bo_results.json.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(message)s")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file: Path = args.results_file if args.results_file is not None else output_dir / "bo_results.json"

    objectives = [rosenbrock(dim=args.rosenbrock_dim), hartmann()]

    def budget_for(obj: Objective) -> tuple[int, int]:
        n_init, n_iters = default_budget(obj.dim)
        if args.n_init is not None:
            n_init = args.n_init
        if args.n_iterations is not None:
            n_iters = args.n_iterations
        return n_init, n_iters

    results = run_grid(
        objectives,
        args.surrogates,
        args.seeds,
        budget_for=budget_for,
        beta=args.beta,
        checkpoint_file=results_file,
    )
    _print_table(results)
    logger.info("Wrote %d runs to %s", len(results), results_file)


if __name__ == "__main__":
    main()
