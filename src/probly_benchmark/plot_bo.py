"""Plot Bayesian optimization results from a JSON file.

Reads the file produced by ``bayesian_optimization.py`` and draws one panel
per objective showing simple regret as a function of the cumulative
evaluation budget. Each surrogate gets its own line -- mean across seeds,
with a shaded ``mean ± std`` band when more than one seed is present.

Default I/O matches the benchmark::

    uv run python -m probly_benchmark.plot_bo
    # reads <project>/bo_outputs/bo_results.json
    # writes <project>/bo_outputs/bo_results.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from probly_benchmark.bayesian_optimization import DEFAULT_OUTPUT_DIR

# Surrogate -> (linestyle, marker, color-index).
_SURROGATE_STYLE: dict[str, tuple[str, str | None, int]] = {
    "gp": ("-", "o", 0),
    "ensemble": ("--", "s", 1),
}


def _aggregate_curve(
    runs: list[dict], *, log_space: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n_obs, center, lower, upper)`` aligned across seeds.

    For ``log_space=True`` the center is the geometric mean and the band
    is one geometric standard deviation -- i.e., aggregation happens in
    log-space, which is the correct convention for log-y plotting and
    avoids negative band edges when ``std > mean``.

    Trims to the shortest seed's trajectory so all seeds share the same
    x-axis.

    Args:
        runs: List of run dicts (one per seed) for a single
            ``(objective, surrogate)`` group.
        log_space: If True, aggregate regret in log space (geometric mean
            and geometric std). Defaults to False (arithmetic).

    Returns:
        Tuple ``(n_obs, center, lower, upper)`` of 1-D numpy arrays of
        the same length. ``center`` is the across-seed central tendency;
        ``lower`` / ``upper`` give the ``mean ± std`` band.
    """
    min_len = min(len(r["iterations"]) for r in runs)
    regret = np.array([[it["regret"] for it in r["iterations"][:min_len]] for r in runs])
    n_obs = np.array([it["n_obs"] for it in runs[0]["iterations"][:min_len]])
    if log_space:
        eps = max(regret[regret > 0].min() if (regret > 0).any() else 1e-12, 1e-12) * 1e-3
        log_r = np.log(np.maximum(regret, eps))
        log_mean = log_r.mean(axis=0)
        log_std = log_r.std(axis=0)
        return n_obs, np.exp(log_mean), np.exp(log_mean - log_std), np.exp(log_mean + log_std)
    mean_r = regret.mean(axis=0)
    std_r = regret.std(axis=0)
    return n_obs, mean_r, mean_r - std_r, mean_r + std_r


def plot_bo(results_file: Path, output: Path | None = None, *, log_y: bool = True) -> None:
    """Plot regret-vs-budget curves from a BO results JSON file.

    Args:
        results_file: Path to the JSON file produced by
            ``bayesian_optimization.py``.
        output: Path to save the figure. Shows interactively if None.
        log_y: If True (default), use a log y-axis. Regret typically spans
            multiple orders of magnitude so log scale is usually clearer.
    """
    with results_file.open() as f:
        runs = json.load(f)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["objective"], run["surrogate"])].append(run)

    objectives = sorted({obj for obj, _ in grouped})
    n_panels = len(objectives)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for ax, objective in zip(axes[0], objectives, strict=True):
        for (obj_name, surrogate), seeds in sorted(grouped.items()):
            if obj_name != objective:
                continue
            n_obs, center, lower, upper = _aggregate_curve(seeds, log_space=log_y)
            ls, _marker, color_idx = _SURROGATE_STYLE.get(surrogate, ("-", None, len(_SURROGATE_STYLE)))
            color = cmap(color_idx)
            label = f"{surrogate} (n_seeds={len(seeds)})"
            ax.plot(n_obs, center, ls=ls, color=color, label=label, linewidth=1.8)
            if len(seeds) > 1:
                ax.fill_between(n_obs, lower, upper, alpha=0.2, color=color)

        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Simple regret")
        ax.set_title(objective)
        if log_y:
            ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and render the BO regret plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used to resolve default --results-file and --output. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="BO results JSON. Defaults to <output-dir>/bo_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <output-dir>/bo_results.png. Use --show to display interactively instead.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively instead of saving to a PNG.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of log.",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    results_file: Path = args.results_file if args.results_file is not None else output_dir / "bo_results.json"
    output: Path | None = (
        None if args.show else (args.output if args.output is not None else output_dir / "bo_results.png")
    )

    plot_bo(results_file, output, log_y=not args.linear_y)


if __name__ == "__main__":
    main()
