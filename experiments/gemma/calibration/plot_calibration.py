"""Generate publication-quality figures from semantic calibration results.

Loads one or more JSON result files from run_experiment.py and produces
four figures: reliability diagram, confidence scatter, entropy distribution,
and calibration comparison bar chart.

Usage:
    uv run python gemma/calibration/plot_calibration.py \
        --results data/results/run_t05.json data/results/run_t10.json \
        --output data/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.calibration import (
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    average_calibration_error,
    reliability_diagram_data,
)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Publication style
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = ["#2176AE", "#D7263D", "#57A773", "#F49D37"]


def load_results(paths: list[str]) -> list[dict]:
    """Load result files and return list of run dicts."""
    runs = []
    for p in paths:
        with Path(p).open() as f:
            runs.append(json.load(f))
    return runs


def label_for_run(run: dict) -> str:
    """Create a display label from run metadata."""
    t = run["metadata"]["temperature"]
    return f"T={t}"


def _has_llm_judge(run: dict) -> bool:
    """Check if a run contains LLM judge results."""
    if not run["results"]:
        return False
    return "is_correct_discrete_llm" in run["results"][0]


def plot_reliability_diagram(runs: list[dict], output_dir: Path) -> None:
    """Figure 1: Reliability diagram with calibration overlays."""
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4.5), squeeze=False)

    for col, run in enumerate(runs):
        ax = axes[0, col]
        results = run["results"]
        conf = np.array([r["confidence_discrete"] for r in results])
        corr = np.array([float(r["is_correct_discrete"]) for r in results])

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")

        # Uncalibrated
        centers, accs, counts = reliability_diagram_data(conf, corr, n_bins=10)
        mask = counts > 0
        ax.bar(
            centers[mask],
            accs[mask],
            width=0.08,
            alpha=0.3,
            color=COLORS[0],
            label="Uncalibrated",
        )
        ax.plot(centers[mask], accs[mask], "o-", color=COLORS[0], markersize=4)

        # Post-hoc calibrated versions
        calibrators = [
            ("Temperature", TemperatureScaler, COLORS[1]),
            ("Platt", PlattScaler, COLORS[2]),
            ("Isotonic", IsotonicCalibrator, COLORS[3]),
        ]
        for name, cls, color in calibrators:
            cal = cls()
            cal.fit(conf, corr)
            cal_conf = cal.calibrate(conf)
            c_centers, c_accs, c_counts = reliability_diagram_data(
                cal_conf,
                corr,
                n_bins=10,
            )
            c_mask = c_counts > 0
            ax.plot(
                c_centers[c_mask],
                c_accs[c_mask],
                "s--",
                color=color,
                markersize=3,
                alpha=0.7,
                label=name,
            )

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram ({label_for_run(run)})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, output_dir, "reliability_diagram")


def plot_confidence_scatter(runs: list[dict], output_dir: Path) -> None:
    """Figure 2: Confidence vs correctness scatter."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, run in enumerate(runs):
        results = run["results"]
        conf = np.array([r["confidence_discrete"] for r in results])
        corr = np.array([float(r["is_correct_discrete"]) for r in results])

        # Jitter correctness for visibility
        rng = np.random.default_rng(seed=42 + i)
        jitter = rng.uniform(-0.04, 0.04, size=len(corr))

        ax.scatter(
            conf,
            corr + jitter,
            alpha=0.4,
            s=15,
            color=COLORS[i % len(COLORS)],
            label=label_for_run(run),
        )

        # Logistic regression trend line
        if len(conf) > 1:
            _logits = np.log(np.clip(conf, 1e-7, 1 - 1e-7) / (1 - np.clip(conf, 1e-7, 1 - 1e-7)))

            def _neg_ll(params: np.ndarray, lg: np.ndarray = _logits, cr: np.ndarray = corr) -> float:
                a, b = params
                p = expit(a * lg + b)
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return -float(np.mean(cr * np.log(p) + (1 - cr) * np.log(1 - p)))

            res = minimize(_neg_ll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
            a, b = res.x
            x_line = np.linspace(conf.min(), conf.max(), 100)
            logits_line = np.log(
                np.clip(x_line, 1e-7, 1 - 1e-7) / (1 - np.clip(x_line, 1e-7, 1 - 1e-7)),
            )
            y_line = expit(a * logits_line + b)
            ax.plot(
                x_line,
                y_line,
                "-",
                color=COLORS[i % len(COLORS)],
                alpha=0.8,
                linewidth=2,
            )

    ax.set_xlabel("Semantic Confidence (discrete)")
    ax.set_ylabel("Correct (0/1, jittered)")
    ax.set_title("Confidence vs Correctness")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.15, 1.15)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "confidence_scatter")


def plot_entropy_distribution(runs: list[dict], output_dir: Path) -> None:
    """Figure 3: Semantic entropy split by correct/incorrect."""
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4), squeeze=False)

    for col, run in enumerate(runs):
        ax = axes[0, col]
        results = run["results"]
        entropy = np.array([r["entropy_discrete"] for r in results])
        corr = np.array([r["is_correct_discrete"] for r in results])

        ax.hist(
            entropy[corr.astype(bool)],
            bins=15,
            alpha=0.6,
            color="#57A773",
            label="Correct",
            density=True,
        )
        ax.hist(
            entropy[~corr.astype(bool)],
            bins=15,
            alpha=0.6,
            color="#D7263D",
            label="Incorrect",
            density=True,
        )

        ax.set_xlabel("Semantic Entropy (discrete)")
        ax.set_ylabel("Density")
        ax.set_title(f"Entropy Distribution ({label_for_run(run)})")
        ax.legend()

    fig.tight_layout()
    _save(fig, output_dir, "entropy_distribution")


def plot_calibration_comparison(runs: list[dict], output_dir: Path) -> None:
    """Figure 4: ACE comparison bar chart across calibration methods."""
    methods = ["Uncalibrated", "Temperature", "Platt", "Isotonic"]
    calibrator_classes = [None, TemperatureScaler, PlattScaler, IsotonicCalibrator]

    n_runs = len(runs)
    x = np.arange(len(methods))
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, run in enumerate(runs):
        results = run["results"]
        conf = np.array([r["confidence_discrete"] for r in results])
        corr = np.array([float(r["is_correct_discrete"]) for r in results])

        aces = []
        for cls in calibrator_classes:
            if cls is None:
                aces.append(float(average_calibration_error(conf, corr)))
            else:
                cal = cls()
                cal.fit(conf, corr)
                cal_conf = cal.calibrate(conf)
                aces.append(float(average_calibration_error(cal_conf, corr)))

        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(
            x + offset,
            aces,
            width,
            label=label_for_run(run),
            color=COLORS[i % len(COLORS)],
            alpha=0.8,
        )

    ax.set_xlabel("Calibration Method")
    ax.set_ylabel("ACE (lower is better)")
    ax.set_title("Calibration Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "calibration_comparison")


def plot_judge_agreement(runs: list[dict], output_dir: Path) -> None:
    """NLI vs LLM judge agreement confusion matrix."""
    judged = [r for r in runs if _has_llm_judge(r)]
    if not judged:
        return

    n = len(judged)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)

    for col, run in enumerate(judged):
        ax = axes[0, col]
        results = run["results"]
        nli = np.array([r["is_correct_discrete"] for r in results])
        llm = np.array([r["is_correct_discrete_llm"] for r in results])

        both_correct = int(np.sum(nli & llm))
        nli_only = int(np.sum(nli & ~llm))
        llm_only = int(np.sum(~nli & llm))
        both_wrong = int(np.sum(~nli & ~llm))
        matrix = np.array([[both_correct, nli_only], [llm_only, both_wrong]])

        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        for (j, k), val in np.ndenumerate(matrix):
            ax.text(
                k,
                j,
                str(val),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if val > matrix.max() / 2 else "black",
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["LLM Correct", "LLM Incorrect"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["NLI Correct", "NLI Incorrect"])
        ax.set_title(f"Judge Agreement ({label_for_run(run)})")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    _save(fig, output_dir, "judge_agreement")


def plot_reliability_comparison(runs: list[dict], output_dir: Path) -> None:
    """Side-by-side reliability diagrams -- NLI vs LLM judge."""
    judged = [r for r in runs if _has_llm_judge(r)]
    if not judged:
        return

    n = len(judged)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.5 * n), squeeze=False)

    for row, run in enumerate(judged):
        results = run["results"]
        conf = np.array([r["confidence_discrete"] for r in results])

        for panel, (corr_key, title_suffix) in enumerate(
            [
                ("is_correct_discrete", "NLI Judge"),
                ("is_correct_discrete_llm", "LLM Judge"),
            ]
        ):
            ax = axes[row, panel]
            corr = np.array([float(r[corr_key]) for r in results])

            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")

            centers, accs, counts = reliability_diagram_data(conf, corr, n_bins=10)
            mask = counts > 0
            ax.bar(
                centers[mask],
                accs[mask],
                width=0.08,
                alpha=0.3,
                color=COLORS[0],
                label="Uncalibrated",
            )
            ax.plot(centers[mask], accs[mask], "o-", color=COLORS[0], markersize=4)

            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{title_suffix} ({label_for_run(run)})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, output_dir, "reliability_comparison")


def plot_calibration_comparison_llm(runs: list[dict], output_dir: Path) -> None:
    """ACE bar chart with both NLI and LLM judge variants."""
    judged = [r for r in runs if _has_llm_judge(r)]
    if not judged:
        return

    methods = ["Uncalibrated", "Temperature", "Platt", "Isotonic"]
    calibrator_classes = [None, TemperatureScaler, PlattScaler, IsotonicCalibrator]

    n_groups = len(methods)
    n_bars = len(judged) * 2
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bar_idx = 0
    for run in judged:
        results = run["results"]
        conf = np.array([r["confidence_discrete"] for r in results])

        for judge_name, corr_key in [
            ("NLI", "is_correct_discrete"),
            ("LLM", "is_correct_discrete_llm"),
        ]:
            corr = np.array([float(r[corr_key]) for r in results])
            aces = []
            for cls in calibrator_classes:
                if cls is None:
                    aces.append(float(average_calibration_error(conf, corr)))
                else:
                    cal = cls()
                    cal.fit(conf, corr)
                    cal_conf = cal.calibrate(conf)
                    aces.append(float(average_calibration_error(cal_conf, corr)))

            offset = (bar_idx - (n_bars - 1) / 2) * width
            color_idx = bar_idx % len(COLORS)
            ax.bar(
                x + offset,
                aces,
                width,
                label=f"{label_for_run(run)} ({judge_name})",
                color=COLORS[color_idx],
                alpha=0.8,
            )
            bar_idx += 1

    ax.set_xlabel("Calibration Method")
    ax.set_ylabel("ACE (lower is better)")
    ax.set_title("Calibration Comparison: NLI vs LLM Judge")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "calibration_comparison_llm")


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as both PDF and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.pdf / {name}.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more JSON result files from run_experiment.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/figures",
        help="Output directory for figures (default: data/figures).",
    )
    return parser.parse_args()


def main() -> None:
    """Load results and generate all figures."""
    args = parse_args()
    output_dir = Path(args.output)

    print(f"Loading {len(args.results)} result file(s)...")
    runs = load_results(args.results)
    for run in runs:
        label = label_for_run(run)
        n = len(run["results"])
        print(f"  {label}: {n} questions")

    print(f"\nGenerating figures in {output_dir}/")
    plot_reliability_diagram(runs, output_dir)
    plot_confidence_scatter(runs, output_dir)
    plot_entropy_distribution(runs, output_dir)
    plot_calibration_comparison(runs, output_dir)

    # LLM judge comparison plots (only generated if judge data present)
    plot_judge_agreement(runs, output_dir)
    plot_reliability_comparison(runs, output_dir)
    plot_calibration_comparison_llm(runs, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
