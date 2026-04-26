"""Plot active learning results from the shared local JSON file.

The shared file is produced by :mod:`probly_benchmark.active_learning`: each
run appends its result dict to a single JSON list, keyed by ``(method,
dataset, strategy, seed)`` so re-runs overwrite previous entries. This
plotter reads that list, groups by ``(method, strategy)``, and plots
mean +/- std across seeds. This is rough local tooling — the production
path goes through wandb.

Usage:
    # Plot accuracy from the default file (al_results.json in cwd)
    uv run python -m probly_benchmark.plot.plot_al

    # Plot ECE instead
    uv run python -m probly_benchmark.plot.plot_al --metric=ece

    # Use a different results file
    uv run python -m probly_benchmark.plot.plot_al --file=runs/al_results.json

    # Save to file
    uv run python -m probly_benchmark.plot.plot_al --output=al_curves.png

    # Restrict to a subset of methods or strategies (comma-separated)
    uv run python -m probly_benchmark.plot.plot_al --methods=dropout,evidential_classification --output=tier1.png
    uv run python -m probly_benchmark.plot.plot_al --strategies=margin,badge,uncertainty --output=strats.png
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load the shared JSON file (a list of run dicts)."""
    return json.loads(Path(path).read_text())


def _group_key(run: dict[str, Any]) -> str:
    """Create a grouping key from method + strategy."""
    return f"{run['method']} / {run['strategy']}"


def plot_learning_curves(
    results_file: str | Path,
    *,
    metric: str = "accuracy",
    output: str | Path | None = None,
    methods: set[str] | None = None,
    strategies: set[str] | None = None,
) -> None:
    """Plot AL learning curves, aggregating over seeds when multiple runs share a config.

    Args:
        results_file: Path to the shared JSON file containing a list of run dicts.
        metric: Which metric to plot ("accuracy" or "ece").
        output: Save figure to this path. Shows interactively if None.
        methods: If given, keep only runs whose ``method`` is in this set.
        strategies: If given, keep only runs whose ``strategy`` is in this set.
    """
    runs = load_results(results_file)

    # Group runs by (method, strategy), applying optional filters
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if methods is not None and run["method"] not in methods:
            continue
        if strategies is not None and run["strategy"] not in strategies:
            continue
        groups[_group_key(run)].append(run)
    if not groups:
        msg = "No results match the given --methods / --strategies filters."
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, group_runs in sorted(groups.items()):
        # Collect per-seed curves (all should have the same labeled_size steps)
        all_x = [it["labeled_size"] for it in group_runs[0]["iterations"]]
        all_y = np.array([[it[metric] for it in r["iterations"]] for r in group_runs])
        mean_y = all_y.mean(axis=0)
        naucs = [r["nauc"] for r in group_runs]
        mean_nauc = np.mean(naucs)

        if len(group_runs) > 1:
            std_y = all_y.std(axis=0)
            ax.fill_between(all_x, mean_y - std_y, mean_y + std_y, alpha=0.15)
            full_label = f"{label} (NAUC={mean_nauc:.3f}, n={len(group_runs)})"
        else:
            full_label = f"{label} (NAUC={mean_nauc:.3f})"

        ax.plot(all_x, mean_y, marker="o", markersize=3, linewidth=1.5, label=full_label)

    ax.set_xlabel("Labeled samples")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Active Learning Curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    metric = "accuracy"
    output: str | None = None
    results_file = "al_results.json"
    methods: set[str] | None = None
    strategies: set[str] | None = None

    for arg in sys.argv[1:]:
        if arg.startswith("--metric="):
            metric = arg.split("=", 1)[1]
        elif arg.startswith("--output="):
            output = arg.split("=", 1)[1]
        elif arg.startswith("--file="):
            results_file = arg.split("=", 1)[1]
        elif arg.startswith("--methods="):
            methods = {m.strip() for m in arg.split("=", 1)[1].split(",") if m.strip()}
        elif arg.startswith("--strategies="):
            strategies = {s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()}
        elif arg == "--help":
            print(__doc__)
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}\n\n{__doc__}")
            sys.exit(1)

    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        sys.exit(1)

    plot_learning_curves(
        results_file,
        metric=metric,
        output=output,
        methods=methods,
        strategies=strategies,
    )
