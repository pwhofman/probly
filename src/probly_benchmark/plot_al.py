"""Plot active learning results from a JSON file."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Strategy -> (linestyle, marker)
_STRATEGY_STYLE = {
    "uncertainty": ("-", None),
    "random": ("--", "o"),
    "margin": ("-.", "s"),
    "badge": (":", "D"),
}


def plot_al(results_file: str, output: str | None = None) -> None:
    """Plot AL learning curves from a JSON results file.

    Args:
        results_file: Path to the JSON file produced by active_learning.py.
        output: Optional path to save the figure. Shows interactively if None.
    """
    with Path(results_file).open() as f:
        runs = json.load(f)

    # Group by (method, strategy) — average over seeds
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["method"], run["strategy"])].append(run)

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")
    methods = sorted({m for m, _ in grouped})
    method_color = {m: cmap(i) for i, m in enumerate(methods)}

    for (method, strategy), seeds in sorted(grouped.items()):
        # Stack iterations across seeds: (n_seeds, n_iters)
        all_acc = np.array([[it["accuracy"] for it in run["iterations"]] for run in seeds])
        x = np.array([it["labeled_size"] for it in seeds[0]["iterations"]])
        mean_acc = all_acc.mean(axis=0)
        nauc = np.mean([r["nauc"] for r in seeds])

        ls, marker = _STRATEGY_STYLE.get(strategy, ("-", None))
        label = f"{method} / {strategy}  (NAUC={nauc:.3f})"
        ax.plot(x, mean_acc, ls=ls, marker=marker, markersize=4, color=method_color[method], label=label)

        if all_acc.shape[0] > 1:
            std_acc = all_acc.std(axis=0)
            ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color=method_color[method])

    ax.set_xlabel("Labeled samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title(Path(results_file).stem.replace("_", " ").title())
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    results = sys.argv[1] if len(sys.argv) > 1 else "al_results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    plot_al(results, out)
