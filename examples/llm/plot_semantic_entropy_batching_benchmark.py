"""Plot semantic entropy batching benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_COLOR = "#1e88e5"
PROBLY_COLOR = "#ff3366"
GRID_COLOR = "#e7e7e7"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
GENERATION_RESULTS_PATH = RESULTS_DIR / "semantic_entropy_generation_batching.json"
NLI_RESULTS_PATH = RESULTS_DIR / "semantic_entropy_nli_batching.json"
GENERATION_PLOT_PATH = RESULTS_DIR / "semantic_entropy_generation_batching.png"
NLI_PLOT_PATH = RESULTS_DIR / "semantic_entropy_nli_batching.png"


def configure_plot_style() -> None:
    """Configure plots to match the existing example figure style."""
    plt.rcParams.update(
        {
            "axes.edgecolor": "#222222",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "axes.titlesize": 12,
            "figure.dpi": 120,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load benchmark rows from a JSON result file.

    Args:
        path: Path to a benchmark JSON file.

    Returns:
        Result rows stored under the ``results`` key.
    """
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    results = payload.get("results")
    if not isinstance(results, list):
        msg = f"Expected {path} to contain a list under the 'results' key."
        raise ValueError(msg)
    return results


def plot_grouped_runtime(
    rows: list[dict[str, Any]],
    *,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """Create a grouped runtime bar plot with speedup annotations.

    Args:
        rows: Benchmark result rows.
        title: Plot title.
        xlabel: X-axis label.
        output_path: Destination image path.
    """
    sample_counts = [int(row["num_samples"]) for row in rows]
    unbatched = [float(row["unbatched_seconds"]) for row in rows]
    batched = [float(row["batched_seconds"]) for row in rows]
    speedups = [row.get("speedup") for row in rows]

    positions = np.arange(len(rows))
    width = 0.38

    _, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(positions - width / 2, unbatched, width, label="Reference", color=REFERENCE_COLOR)
    axis.bar(positions + width / 2, batched, width, label="probly", color=PROBLY_COLOR)

    ymax = max([*unbatched, *batched], default=1.0)
    offset = ymax * 0.03 if ymax > 0 else 0.01
    for position, unbatched_seconds, batched_seconds, speedup in zip(
        positions,
        unbatched,
        batched,
        speedups,
        strict=True,
    ):
        if isinstance(speedup, int | float):
            label = f"{speedup:.2f}x"
        else:
            label = "n/a"
        axis.text(
            position,
            max(unbatched_seconds, batched_seconds) + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_title(title, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Runtime (seconds)")
    axis.set_xticks(positions, [str(value) for value in sample_counts])
    axis.legend(loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.13))
    axis.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_ylim(top=ymax * 1.18 if ymax > 0 else 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    """Read benchmark JSON files and generate runtime plots."""
    configure_plot_style()
    generation_rows = load_results(GENERATION_RESULTS_PATH)
    nli_rows = load_results(NLI_RESULTS_PATH)

    plot_grouped_runtime(
        generation_rows,
        title="Semantic Entropy Sampling Runtime",
        xlabel="Generated samples per question",
        output_path=GENERATION_PLOT_PATH,
    )
    print(f"Wrote generation plot to {GENERATION_PLOT_PATH}")

    plot_grouped_runtime(
        nli_rows,
        title="Semantic Entropy NLI Runtime",
        xlabel="Clustered samples",
        output_path=NLI_PLOT_PATH,
    )
    print(f"Wrote NLI plot to {NLI_PLOT_PATH}")


if __name__ == "__main__":
    main()
