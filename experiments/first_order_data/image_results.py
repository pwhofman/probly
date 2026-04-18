from __future__ import annotations

from collections.abc import Iterator
import csv
import json
from pathlib import Path
import sys

from first_order_data.utils import coverage_convex_hull_relaxed, coverage_convex_hull
import numpy as np

from probly.representation.credal_set.array import ArrayProbabilityIntervalsCredalSet
from probly.representation.distribution import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)

UNCERTAINTY_COLUMNS = ("total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prediction_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    fieldnames = reader.fieldnames or []
    target_columns = [name for name in fieldnames if name.startswith("target::")]
    prediction_columns = [name for name in fieldnames if name.startswith("pred::")]

    targets = np.array([[float(row[name]) for name in target_columns] for row in rows], dtype=float)
    predictions = np.array([[float(row[name]) for name in prediction_columns] for row in rows], dtype=float)
    uncertainty = {
        name: np.array([float(row[name]) for row in rows], dtype=float)
        for name in UNCERTAINTY_COLUMNS
        if name in fieldnames
    }
    return targets, predictions, uncertainty


def interval_coverage(member_probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Credal-set interval coverage via :class:`ArrayProbabilityIntervalsCredalSet`.

    `member_probabilities` has shape `(n_instances, n_members, n_classes)`.
    Credal set is built from per-class min/max across members.
    `contains` tests whether each target lies inside every class interval.
    """
    members_first = np.moveaxis(member_probabilities, 1, 0)  # (n_members, n_instances, n_classes)
    sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(members_first),
        sample_axis=0,
    )
    credal_set = ArrayProbabilityIntervalsCredalSet.from_array_sample(sample)
    return float(credal_set.contains(targets).mean())


def convex_hull_coverage_relaxed(member_probabilities: np.ndarray, targets: np.ndarray, epsilon: float) -> float:
    """Relaxed convex-hull coverage, where a target is considered covered if it's within `epsilon` of the convex hull."""
    return float(coverage_convex_hull_relaxed(member_probabilities, targets, epsilon=epsilon))


def total_variation_distance(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(0.5 * np.abs(predictions - targets).sum(axis=1)))


def dataset_row(run_dir: Path, dataset_result: dict) -> dict[str, float | str]:
    fold_rows = []

    for fold_result in dataset_result["folds"]:
        ensemble_targets, ensemble_predictions, ensemble_uncertainty = load_prediction_csv(
            run_dir
            / dataset_result["dataset_name"]
            / dataset_result["encoder_name"]
            / fold_result["test_fold"]
            / "ensemble_predictions.csv"
        )
        member_probabilities = np.stack(
            [load_prediction_csv(Path(path))[1] for path in fold_result["member_prediction_files"]],
            axis=1,
        )

        fold_rows.append(
            {
                "member_ce": float(np.mean(fold_result["member_cross_entropies"])),
                "ensemble_ce": float(fold_result["ensemble_cross_entropy"]),
                "interval_coverage": interval_coverage(member_probabilities, ensemble_targets),
                "convex_hull_coverage": float(coverage_convex_hull(member_probabilities, ensemble_targets)),
                "tv_distance": total_variation_distance(ensemble_predictions, ensemble_targets),
                **{name: float(ensemble_uncertainty[name].mean()) for name in UNCERTAINTY_COLUMNS},
            }
        )

    keys = (
        "member_ce",
        "ensemble_ce",
        "interval_coverage",
        "convex_hull_coverage",
        "tv_distance",
        *UNCERTAINTY_COLUMNS,
    )
    return {
        "dataset_name": dataset_result["dataset_name"],
        **{key: float(np.mean([row[key] for row in fold_rows])) for key in keys},
    }


def iter_dataset_rows(run_dir: Path) -> Iterator[dict[str, float | str]]:
    results = load_json(run_dir / "results.json")
    for dataset_result in results:
        yield dataset_row(run_dir, dataset_result)


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&")


def get_latex_table(run_dir: Path) -> str:
    rows = list(iter_dataset_rows(run_dir))
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Dataset & Member CE & Ensemble CE & Interval Cov. & Hull Cov. & TV Distance & TU & AU & EU \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(
            f"{latex_escape(str(row['dataset_name']))} & "
            f"{row['member_ce']:.4f} & "
            f"{row['ensemble_ce']:.4f} & "
            f"{row['interval_coverage']:.4f} & "
            f"{row['convex_hull_coverage']:.4f} & "
            f"{row['tv_distance']:.4f} & "
            f"{row['total_uncertainty']:.4f} & "
            f"{row['aleatoric_uncertainty']:.4f} & "
            f"{row['epistemic_uncertainty']:.4f} \\\\"
        )

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main():
    run_dir = Path(sys.argv[1])

    for row in iter_dataset_rows(run_dir):
        print(
            row["dataset_name"] + ":",
            "\n  Member CE:",
            f"{row['member_ce']:.4f}",
            "\n  Ensemble CE:",
            f"{row['ensemble_ce']:.4f}",
            "\n  Interval Coverage:",
            f"{row['interval_coverage']:.4f}",
            "\n  Convex Hull Coverage:",
            f"{row['convex_hull_coverage']:.4f}",
            "\n  TV Distance:",
            f"{row['tv_distance']:.4f}",
            "\n  Total Uncertainty:",
            f"{row['total_uncertainty']:.4f}",
            "\n  Aleatoric Uncertainty:",
            f"{row['aleatoric_uncertainty']:.4f}",
            "\n  Epistemic Uncertainty:",
            f"{row['epistemic_uncertainty']:.4f}",
        )

    print()
    print(get_latex_table(run_dir))


if __name__ == "__main__":
    main()
