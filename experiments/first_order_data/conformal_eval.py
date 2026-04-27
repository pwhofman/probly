"""Evaluate conformal prediction strategies on cached first-order results."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from summarize_dcic_ensemble_results import (
    load_json,
    load_prediction_csv,
)
import numpy as np
import numpy.typing as npt
from torch import nn

from probly.calibrator import calibrate
from probly.metrics._common import average_set_size, empirical_coverage_classification
from probly.method.conformal import conformal_aps, conformal_lac, conformal_raps
from probly.representer import representer

type FloatArray = npt.NDArray[np.floating]
type IntArray = npt.NDArray[np.integer]
type BoolArray = npt.NDArray[np.bool_]
type UncertaintyValues = Mapping[str, FloatArray]
type MetricRow = tuple[float, float, float]  # (hard_coverage, soft_coverage, avg_set_size)
type ResultRows = dict[tuple[str, str], list[MetricRow]]


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


def make_predictor(name: str, model: CachedProbsModel) -> object:
    if name == "lac":
        return conformal_lac(model)
    if name == "aps":
        return conformal_aps(model, randomized=True)
    if name == "raps":
        return conformal_raps(model, randomized=True)
    msg = f"Unknown method '{name}'"
    raise ValueError(msg)


def soft_empirical_coverage(prediction_sets: BoolArray, targets_soft: FloatArray) -> float:
    """Compute the target probability mass covered by each prediction set."""
    return float(np.mean(np.sum(prediction_sets * targets_soft, axis=1)))


def fold_coverage(
    probs: FloatArray,
    targets_soft: FloatArray,
    uncertainty: UncertaintyValues,
    args: argparse.Namespace,
) -> ResultRows:
    """runs every (method, strategy) combination over a set amount random cal/test splits.

    Returns a dict mapping (method, strategy) to per-split metric tuples.
    """
    if args.label_mode == "sample":
        # draw one label per row from the soft target distribution
        label_rng = np.random.default_rng(args.seed)
        cum = targets_soft.cumsum(axis=1)
        u = label_rng.random(len(targets_soft))[:, None]
        y_hard = (u < cum).argmax(axis=1)
    else:
        y_hard = targets_soft.argmax(axis=1)
    model = CachedProbsModel(probs)
    n_samples = len(y_hard)
    rng = np.random.default_rng(args.seed)

    rows: ResultRows = {(method, strategy): [] for method in args.methods for strategy in args.conditioning}
    for _ in range(args.num_splits):
        seed = int(rng.integers(0, 2**31 - 1))
        perm = np.random.default_rng(seed).permutation(n_samples)
        split_at = n_samples // 2
        idx_cal, idx_test = perm[:split_at], perm[split_at:]
        y_cal, y_test = y_hard[idx_cal], y_hard[idx_test]
        targets_soft_test = targets_soft[idx_test]

        for method in args.methods:
            predictor = make_predictor(method, model)
            calibrated = calibrate(predictor, args.alpha, y_cal, idx_cal)
            sets = np.asarray(representer(calibrated).predict(idx_test).array, dtype=bool)
            rows[(method, "split")].append(
                (
                    float(empirical_coverage_classification(sets, y_test)),
                    soft_empirical_coverage(sets, targets_soft_test),
                    float(average_set_size(sets)),
                )
                    )
    return rows


def print_summary(fold_name: str, rows: ResultRows) -> None:
    headers = ("hard_cover", "soft_cover", "size")
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
    parser.add_argument("--methods", nargs="+", choices=("lac", "aps", "raps"), default=("lac", "aps", "raps"))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
