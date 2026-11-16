"""Plots for the benchmark (PIPE): metric vs model size, and rejection curves.

Run from ``experiments/llm-probly`` after ``run_benchmark`` (for each model)::

    uv run python -m scripts.plot_results --dataset triviaqa --n 200 --curves-model qwen1.5b

Writes into ``results/plots/``:

- ``<dataset>_<n>_auroc_vs_size.png`` and ``..._prr_vs_size.png``: one line per
  method, x = model size, error bars = bootstrap 95 % interval (AUROC only).
- ``<dataset>_<n>_<model>_rejection.png``: accuracy of the kept questions when
  the most uncertain ones are rejected, one line per method, plus the oracle
  (perfect ranking) and random (flat at the accuracy).

Every method keeps the same color and marker in every plot (``STYLE`` below),
so a method is recognisable across figures. The numbers behind the lines are in
``results/comparison_<dataset>_<n>.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

from common.data import cache_path, load_jsonl
from common.evaluate import kept_accuracy_curve
from common.io import RESULTS_ROOT
from common.summary import compare_models, find_models

# Fixed color + marker per method: color follows the method, never its position in a plot.
# Colors: the 8 categorical slots of the shared chart palette (checked for colour-blind
# separation); markers are a second cue, so lines can be told apart without colour too.
STYLE = {
    "likelihood_mean": ("#2a78d6", "o"),
    "p_true_zeroshot": ("#eb6834", "s"),
    "p_true_brainstorm": ("#1baf7a", "D"),
    "p_true_empirical": ("#eda100", "^"),
    "verbalized_2s": ("#e87ba4", "v"),
    "semantic_entropy": ("#008300", "P"),
    "discrete_semantic_entropy": ("#6250d6", "X"),
    "likelihood_sum": ("#e34948", "h"),
}
OTHER = ("#8a8984", ".")  # any further method: gray, listed in the legend by name
HIDDEN = {"dummy_random"}  # test plugin, not a real method
DATASET_LABEL = {"triviaqa": "TriviaQA", "mmlu": "MMLU"}
SIZE_LABEL = {"smollm135m": "135M", "qwen0.5b": "0.5B", "qwen1.5b": "1.5B", "qwen3b": "3B", "qwen7b": "7B"}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def style_axes(ax) -> None:
    """Quiet grid and axes, so the data lines stand out."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def legend(ax) -> None:
    """Legend outside the plot on the right, text in ink (not series colours)."""
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK)


def plot_vs_size(long, metric: str, title: str, out: Path) -> None:
    """One line per method: the metric at each model size."""
    models = list(dict.fromkeys(long["model"]))
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    methods = sorted(set(long["method"]) - HIDDEN)
    for k, method in enumerate(methods):
        part = long[long["method"] == method].set_index("model").reindex(models)
        color, marker = STYLE.get(method, OTHER)
        y = part[metric].to_numpy(dtype=float)
        shift = (k - (len(methods) - 1) / 2) * 0.025  # small sideways offset so error bars don't overlap
        if metric == "auroc":  # bootstrap 95 % interval as thin error bars
            err = np.vstack([y - part["ci_low"].to_numpy(float), part["ci_high"].to_numpy(float) - y])
            ax.errorbar(x + shift, y, yerr=err, color=color, linewidth=0, elinewidth=1, capsize=3, alpha=0.6)
        ax.plot(x + shift, y, color=color, marker=marker, markersize=8, linewidth=2, label=method,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    reference = 0.5 if metric == "auroc" else 0.0
    ax.axhline(reference, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate("random", xy=(0.01, reference), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", color=INK_2, fontsize=8)
    ax.set_xticks(x, [SIZE_LABEL.get(m, m) for m in models])
    ax.set_xlabel("model size (Qwen2.5-Instruct)", color=INK_2, fontsize=9)
    ax.set_ylabel(metric.upper(), color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style_axes(ax)
    legend(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def load_method_scores(folder: Path) -> dict[str, dict]:
    """All score files of one run (skips summary.md and anything that is not a score record)."""
    scores = {}
    for file in sorted(folder.glob("*.json")):
        record = json.loads(file.read_text())
        if "uncertainty" in record and file.stem not in HIDDEN:
            scores[file.stem] = record
    return scores


def plot_rejection(rows: list[dict], folder: Path, title: str, out: Path) -> None:
    """Accuracy of the kept questions vs fraction rejected, per method + oracle + random."""
    correct_by_id = {str(r["id"]): bool(r["correct"]) for r in rows}
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)

    correct_all = np.array(list(correct_by_id.values()))
    rejected, oracle = kept_accuracy_curve(~correct_all, correct_all)  # perfect: wrong answers first
    ax.plot(rejected, oracle, color=INK, linewidth=1.5, linestyle="--", label="oracle (perfect)")
    ax.axhline(correct_all.mean(), color=INK_2, linewidth=1, linestyle=":", label="random")

    for method, record in load_method_scores(folder).items():
        correct = [correct_by_id[str(i)] for i in record["ids"]]
        rejected, accuracy = kept_accuracy_curve(record["uncertainty"], correct)
        color, marker = STYLE.get(method, OTHER)
        keep = rejected <= 0.9  # the last points average only a few questions: too noisy
        ax.plot(rejected[keep], accuracy[keep], color=color, linewidth=2, label=method,
                marker=marker, markevery=max(1, keep.sum() // 6), markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_xlabel("most uncertain questions rejected", color=INK_2, fontsize=9)
    ax.set_ylabel("accuracy on the questions kept", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style_axes(ax)
    legend(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200, help="number of questions")
    parser.add_argument("--curves-model", default="qwen1.5b", help="model for the rejection curves")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap samples for the AUROC interval")
    return parser.parse_args()


def main() -> None:
    """Make all plots for one dataset."""
    args = parse_args()
    run_name = f"{args.dataset}_{args.n}"
    models = find_models(run_name)
    if not models:
        raise SystemExit(f"no results/<model>/{run_name}/ folders yet; run run_benchmark first")
    out_dir = RESULTS_ROOT / "plots"
    out_dir.mkdir(exist_ok=True)

    rows_by_model = {m: load_jsonl(cache_path(m, args.n, args.dataset)) for m in models}
    long = compare_models(rows_by_model, run_name, n_boot=args.n_boot)
    name = f"{DATASET_LABEL.get(args.dataset, args.dataset)}, {args.n} questions"
    written = []
    for metric in ("auroc", "prr"):
        out = out_dir / f"{run_name}_{metric}_vs_size.png"
        plot_vs_size(long, metric, f"{metric.upper()} by model size · {name}", out)
        written.append(out)

    if args.curves_model in rows_by_model:
        out = out_dir / f"{run_name}_{args.curves_model}_rejection.png"
        label = SIZE_LABEL.get(args.curves_model, args.curves_model)
        plot_rejection(rows_by_model[args.curves_model], RESULTS_ROOT / args.curves_model / run_name,
                       f"Rejection curves · Qwen {label} · {name}", out)
        written.append(out)
    else:
        print(f"no results for {args.curves_model}, rejection curves skipped")

    for path in written:
        print("saved", path.relative_to(RESULTS_ROOT.parent))


if __name__ == "__main__":
    main()
