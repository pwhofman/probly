from __future__ import annotations

import sys
import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from probly.metrics import coverage_convex_hull


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prediction_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    target_columns = [name for name in reader.fieldnames or [] if name.startswith("target::")]
    prediction_columns = [name for name in reader.fieldnames or [] if name.startswith("pred::")]

    targets = np.array(
        [[float(row[name]) for name in target_columns] for row in rows],
        dtype=float,
    )
    predictions = np.array(
        [[float(row[name]) for name in prediction_columns] for row in rows],
        dtype=float,
    )
    return targets, predictions


def interval_coverage(member_probabilities: np.ndarray, targets: np.ndarray) -> float:
    lower = member_probabilities.min(axis=1)
    upper = member_probabilities.max(axis=1)
    inside = (targets >= lower) & (targets <= upper)
    return float(np.mean(np.all(inside, axis=1)))


def total_variation_distance(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(0.5 * np.abs(predictions - targets).sum(axis=1)))


def dataset_row(run_dir: Path, dataset_result: dict) -> dict[str, float | str]:
    fold_rows = []

    for fold_result in dataset_result["folds"]:
        ensemble_targets, ensemble_predictions = load_prediction_csv(
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
            }
        )

    return {
        "dataset_name": dataset_result["dataset_name"],
        "member_ce": float(np.mean([row["member_ce"] for row in fold_rows])),
        "ensemble_ce": float(np.mean([row["ensemble_ce"] for row in fold_rows])),
        "interval_coverage": float(np.mean([row["interval_coverage"] for row in fold_rows])),
        "convex_hull_coverage": float(np.mean([row["convex_hull_coverage"] for row in fold_rows])),
        "tv_distance": float(np.mean([row["tv_distance"] for row in fold_rows])),
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
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & Member CE & Ensemble CE & Interval Cov. & Convex Hull Cov. & TV Distance \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(
            f"{latex_escape(str(row['dataset_name']))} & "
            f"{row['member_ce']:.4f} & "
            f"{row['ensemble_ce']:.4f} & "
            f"{row['interval_coverage']:.4f} & "
            f"{row['convex_hull_coverage']:.4f} & "
            f"{row['tv_distance']:.4f} \\\\"
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
        )

    print()
    print(get_latex_table(run_dir))


if __name__ == "__main__":
    main()
