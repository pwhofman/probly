"""Plot active learning results from results.json files.

Auto-discovers result files, aggregates over seeds, and plots mean +/- std.

Usage:
    # Auto-discover all results under outputs/ and plot accuracy
    uv run python -m probly_benchmark.plot.plot_al

    # Plot ECE instead
    uv run python -m probly_benchmark.plot.plot_al --metric=ece

    # Save to file
    uv run python -m probly_benchmark.plot.plot_al --output=al_curves.png

    # Use a different results directory
    uv run python -m probly_benchmark.plot.plot_al --dir=outputs/2026-04-24

    # Restrict to a subset of methods or strategies (comma-separated)
    uv run python -m probly_benchmark.plot.plot_al --methods=dropout,evidential_classification --output=tier1.png
    uv run python -m probly_benchmark.plot.plot_al --strategies=margin,badge,uncertainty --output=strats.png

    # Pass specific files (no aggregation)
    uv run python -m probly_benchmark.plot.plot_al results1.json results2.json
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | Path) -> dict:
    """Load a single results.json file."""
    return json.loads(Path(path).read_text())


def discover_results(root: str | Path = "outputs") -> list[Path]:
    """Find all results.json files under a directory."""
    return sorted(Path(root).rglob("results.json"))


def _group_key(data: dict) -> str:
    """Create a grouping key from method + strategy."""
    return f"{data['method']} / {data['strategy']}"


def plot_learning_curves(
    *result_paths: str | Path,
    metric: str = "accuracy",
    output: str | Path | None = None,
    methods: set[str] | None = None,
    strategies: set[str] | None = None,
) -> None:
    """Plot AL learning curves, aggregating over seeds when multiple runs share a config.

    Args:
        *result_paths: Paths to results.json files.
        metric: Which metric to plot ("accuracy" or "ece").
        output: Save figure to this path. Shows interactively if None.
        methods: If given, keep only runs whose ``method`` is in this set.
        strategies: If given, keep only runs whose ``strategy`` is in this set.
    """
    # Group runs by (method, strategy), applying optional filters
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in result_paths:
        data = load_results(path)
        if methods is not None and data["method"] not in methods:
            continue
        if strategies is not None and data["strategy"] not in strategies:
            continue
        groups[_group_key(data)].append(data)
    if not groups:
        msg = "No results match the given --methods / --strategies filters."
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, runs in sorted(groups.items()):
        # Collect per-seed curves (all should have the same labeled_size steps)
        all_x = [it["labeled_size"] for it in runs[0]["iterations"]]
        all_y = np.array([[it[metric] for it in run["iterations"]] for run in runs])
        mean_y = all_y.mean(axis=0)
        naucs = [run["nauc"] for run in runs]
        mean_nauc = np.mean(naucs)

        if len(runs) > 1:
            std_y = all_y.std(axis=0)
            ax.fill_between(all_x, mean_y - std_y, mean_y + std_y, alpha=0.15)
            full_label = f"{label} (NAUC={mean_nauc:.3f}, n={len(runs)})"
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

    paths: list[str] = []
    metric = "accuracy"
    output = None
    results_dir = "outputs"
    methods: set[str] | None = None
    strategies: set[str] | None = None

    for arg in sys.argv[1:]:
        if arg.startswith("--metric="):
            metric = arg.split("=", 1)[1]
        elif arg.startswith("--output="):
            output = arg.split("=", 1)[1]
        elif arg.startswith("--dir="):
            results_dir = arg.split("=", 1)[1]
        elif arg.startswith("--methods="):
            methods = {m.strip() for m in arg.split("=", 1)[1].split(",") if m.strip()}
        elif arg.startswith("--strategies="):
            strategies = {s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()}
        elif arg == "--help":
            print(__doc__)
            sys.exit(0)
        else:
            paths.append(arg)

    # Auto-discover if no explicit paths given
    if not paths:
        discovered = discover_results(results_dir)
        if not discovered:
            print(f"No results.json found under {results_dir}/")
            sys.exit(1)
        print(f"Found {len(discovered)} result files under {results_dir}/")
        paths = [str(p) for p in discovered]

    plot_learning_curves(*paths, metric=metric, output=output, methods=methods, strategies=strategies)
