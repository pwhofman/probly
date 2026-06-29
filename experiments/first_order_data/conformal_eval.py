"""Evaluate conformal prediction strategies on cached first-order results."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import math
from pathlib import Path

from summarize_dcic_ensemble_results import (
    load_json,
    load_prediction_csv,
)
import numpy as np
import numpy.typing as npt
from torch import nn
import matplotlib.pyplot as plt

from probly.calibrator import calibrate
from probly.metrics import average_set_size, empirical_coverage_classification
from probly.method.conformal import conformal_aps, conformal_lac, conformal_raps, conformal_saps
from probly.representer import representer

type FloatArray = npt.NDArray[np.floating]
type IntArray = npt.NDArray[np.integer]
type BoolArray = npt.NDArray[np.bool_]
type UncertaintyValues = Mapping[str, FloatArray]
type MetricRow = tuple[float, float, float]  # (marginal_coverage, conditional_coverage, avg_set_size)
type ResultRows = dict[tuple[str, str], list[MetricRow]]
type CalibrationCheckRow = tuple[float, float]  # (sample_kl, sample_tv)
type CalibrationSplit = tuple[IntArray, IntArray]  # (calibration_indices, test_indices)


class CachedProbsModel(nn.Module):
    """Caches ensemble probabilities so probly's CP API can work with them without needing to re-run the models."""

    def __init__(self, probs: FloatArray):
        """Stores cached probability rows."""
        super().__init__()
        self.probs = np.asarray(probs, dtype=float)

    def forward(self, x: npt.ArrayLike) -> FloatArray:
        """Return probabilities for cached row indices."""
        return self.predict_proba(x)

    def predict_proba(self, x: npt.ArrayLike) -> FloatArray:
        """Return probabilities for cached row indices."""
        return self.probs[np.asarray(x, dtype=int)]


def make_predictor(name: str, model: CachedProbsModel, args: argparse.Namespace) -> object:
    if name == "lac":
        return conformal_lac(model)
    if name == "aps":
        return conformal_aps(model, randomized=True)
    if name == "saps":
        return conformal_saps(model, randomized=True, lambda_val=args.saps_lambda)
    if name == "raps":
        return conformal_raps(model, randomized=True)
    msg = f"Unknown method '{name}'"
    raise ValueError(msg)


def conditional_coverage(prediction_sets: BoolArray, targets_soft: FloatArray, alpha: float) -> float:
    """Compute the fraction of sets covering at least the target probability mass."""
    return float(np.mean(np.sum(prediction_sets * targets_soft, axis=1) >= 1 - alpha))


def marginal_coverage(prediction_sets: BoolArray, y_true: IntArray) -> float:
    """Compute marginal empirical coverage against hard labels."""
    return float(empirical_coverage_classification(prediction_sets, y_true))


def sample_labels(targets_soft: FloatArray, rng: np.random.Generator) -> IntArray:
    """Draw one hard label per row from soft target probabilities."""
    cum = targets_soft.cumsum(axis=1)
    u = rng.random(len(targets_soft))[:, None]
    return (u < cum).argmax(axis=1)


def calibration_splits(n_samples: int, args: argparse.Namespace) -> list[CalibrationSplit]:
    """Return the splits used by repeated split conformal evaluation."""
    rng = np.random.default_rng(args.seed)
    splits = []
    for _ in range(args.num_splits):
        seed = int(rng.integers(0, 2**31 - 1))
        perm = np.random.default_rng(seed).permutation(n_samples)
        split_at = n_samples // 2
        splits.append((perm[:split_at], perm[split_at:]))
    return splits


def kl_divergence_rows(targets: FloatArray, predictions: FloatArray) -> FloatArray:
    """Computes KL(target || prediction) for each row."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.clip(np.asarray(predictions, dtype=np.float64), 1e-8, 1.0)
    terms = np.zeros_like(targets, dtype=np.float64)
    mask = targets > 0
    terms[mask] = targets[mask] * np.log(targets[mask] / predictions[mask])
    return terms.sum(axis=1)


def calibration_check(
    probs: FloatArray,
    targets_soft: FloatArray,
    args: argparse.Namespace,
) -> tuple[list[CalibrationCheckRow], FloatArray, FloatArray]:
    """Computes per-sample calibration distance between targets and ensemble predictions."""
    calibration_tv_distances = []
    calibration_kl_distances = []
    rows = []

    for idx_cal, _idx_test in calibration_splits(len(targets_soft), args):
        sample_tv = 0.5 * np.abs(targets_soft[idx_cal] - probs[idx_cal]).sum(axis=1)
        sample_kl = kl_divergence_rows(targets_soft[idx_cal], probs[idx_cal])
        calibration_tv_distances.append(sample_tv)
        calibration_kl_distances.append(sample_kl)
        rows.append((float(sample_kl.mean()), float(sample_tv.mean())))

    return rows, np.concatenate(calibration_tv_distances), np.concatenate(calibration_kl_distances)


def print_calibration_summary(fold_name: str, rows: list[CalibrationCheckRow]):
    """Print a compact calibration distribution check table."""
    arr = np.asarray(rows, dtype=float)
    means, stds = arr.mean(axis=0), arr.std(axis=0)
    print(f"\n{fold_name} calibration check:")
    print("  metric          mean         std")
    print(f"  sample KL       {means[0]:<12.4f} {stds[0]:.4f}")
    print(f"  sample TV       {means[1]:<12.4f} {stds[1]:.4f}")


def save_combined_calibration_histogram(
    output_dir: Path,
    rows: list[tuple[str, str, FloatArray]],
    filename: str,
    xlabel: str,
    mean_label: str,
    bins: FloatArray,
    xlim: tuple[float, float] | None = None,
    overflow_at: float | None = None,
) -> Path:
    """Save all dataset calibration histograms in one subplot grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    n_cols = 2
    n_rows = math.ceil(len(rows) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), squeeze=False)

    for ax, (dataset_name, fold_name, values) in zip(axes.ravel(), rows, strict=False):
        weights = np.full_like(values, 100 / len(values), dtype=float)
        mean = float(values.mean())
        ax.hist(values, bins=bins, weights=weights, edgecolor="black")
        ax.axvline(mean, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{dataset_name} / {fold_name}")
        ax.set_ylabel("%")
        ax.text(
            0.98,
            0.92,
            f"{mean_label}={mean:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )
        if overflow_at is not None:
            overflow = float(np.mean(values > overflow_at) * 100)
            if overflow > 0:
                ax.text(
                    0.98,
                    0.78,
                    f">{overflow_at:.2f}: {overflow:.1f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
                )
        if xlim is not None:
            ax.set_xlim(*xlim)

    for ax in axes.ravel()[len(rows) :]:
        ax.axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel(xlabel)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fold_coverage(
    probs: FloatArray,
    targets_soft: FloatArray,
    uncertainty: UncertaintyValues,
    args: argparse.Namespace,
) -> ResultRows:
    """runs every (method, strategy) combination over a set amount random cal/test splits.

    Returns a dict mapping (method, strategy) to per-split metric tuples.
    """
    label_rng = np.random.default_rng(args.seed)
    y_marginal = sample_labels(targets_soft, label_rng)
    if args.label_mode == "sample":
        y_calibration = y_marginal
    else:
        y_calibration = targets_soft.argmax(axis=1)
    model = CachedProbsModel(probs)
    n_samples = len(y_calibration)

    rows: ResultRows = {(method, strategy): [] for method in args.methods for strategy in args.conditioning}
    for idx_cal, idx_test in calibration_splits(n_samples, args):
        y_cal, y_test = y_calibration[idx_cal], y_marginal[idx_test]
        targets_soft_test = targets_soft[idx_test]

        for method in args.methods:
            predictor = make_predictor(method, model, args)
            calibrated = calibrate(predictor, args.alpha, y_cal, idx_cal)
            sets = np.asarray(representer(calibrated).predict(idx_test).array, dtype=bool)
            rows[(method, "split")].append(
                (
                    marginal_coverage(sets, y_test),
                    conditional_coverage(sets, targets_soft_test, alpha=args.alpha),
                    float(average_set_size(sets)),
                )
            )
    return rows


def print_summary(fold_name: str, rows: ResultRows):
    headers = ("marginal", "conditional", "size")
    print(f"\n{fold_name}:")
    print(f"  {'method':<5} {'strategy':<12} " + " ".join(f"{h:<13}" for h in headers))
    for (method, strategy), seed_rows in rows.items():
        arr = np.asarray(seed_rows, dtype=float)
        means, stds = arr.mean(axis=0), arr.std(axis=0)
        cells = " ".join(f"{m:.3f}\u00b1{s:.3f}  " for m, s in zip(means, stds, strict=True))
        print(f"  {method:<5} {strategy:<12} {cells}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)  # miscoverage rate, target coverage = 1 - alpha
    parser.add_argument("--num-splits", type=int, default=20)  # number of CAL/TEST splits to average over
    parser.add_argument("--seed", type=int, default=0)  # seed for the bootstrap RNG
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("lac", "aps", "saps", "raps"),
        default=("lac", "aps", "saps", "raps"),
    )
    parser.add_argument("--saps-lambda", type=float, default=0.1)
    parser.add_argument(
        "--conditioning",
        nargs="+",
        choices=("split",),
        default=("split",),
    )
    parser.add_argument(
        "--label-mode",
        choices=("argmax", "sample"),
        default="argmax",
    )
    parser.add_argument(
        "--calibration-check",
        action="store_true",
        help="Print per-sample KL/TV checks on the calibration rows + histograms.",
    )
    parser.add_argument(
        "--calibration-plot-dir",
        type=Path,
        help="Root directory for calibration histogram saving. Defaults to out/calibration_checks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    calibration_plot_rows_tv = []
    calibration_plot_rows_kl = []

    for dataset_result in load_json(args.run_dir / "results.json"):
        for fold_result in dataset_result["folds"]:
            targets_soft, probs, uncertainty = load_prediction_csv(
                args.run_dir
                / dataset_result["dataset_name"]
                / dataset_result["encoder_name"]
                / fold_result["test_fold"]
                / "ensemble_predictions.csv",
            )
            rows = fold_coverage(probs, targets_soft, uncertainty, args)
            print_summary(f"{dataset_result['dataset_name']} / {fold_result['test_fold']}", rows)
            if args.calibration_check:
                calibration_rows, tv_distances, kl_distances = calibration_check(probs, targets_soft, args)
                print_calibration_summary(
                    f"{dataset_result['dataset_name']} / {fold_result['test_fold']}",
                    calibration_rows,
                )
                calibration_plot_rows_tv.append(
                    (
                        dataset_result["dataset_name"],
                        fold_result["test_fold"],
                        tv_distances,
                    )
                )
                calibration_plot_rows_kl.append(
                    (
                        dataset_result["dataset_name"],
                        fold_result["test_fold"],
                        kl_distances,
                    )
                )

    if args.calibration_check:
        plot_dir = (args.calibration_plot_dir or Path("out/calibration_checks")) / args.run_dir
        tv_path = save_combined_calibration_histogram(
            plot_dir,
            calibration_plot_rows_tv,
            "calibration_tv_hist.png",
            "Per-sample TV(target, ensemble)",
            "mean TV",
            np.arange(0, 1.05, 0.05),
            xlim=(0, 1),
        )
        all_kl_values = np.concatenate([values for _dataset, _fold, values in calibration_plot_rows_kl])
        kl_upper = max(0.05, float(np.ceil(float(np.quantile(all_kl_values, 0.99)) / 0.05) * 0.05))
        kl_path = save_combined_calibration_histogram(
            plot_dir,
            calibration_plot_rows_kl,
            "calibration_kl_hist.png",
            "Per-sample KL(target || ensemble)",
            "mean KL",
            np.arange(0, kl_upper + 0.05, 0.05),
            xlim=(0, kl_upper),
            overflow_at=kl_upper,
        )
        print(f"\nCombined TV histogram: {tv_path}")
        print(f"Combined KL histogram: {kl_path}")


if __name__ == "__main__":
    main()
