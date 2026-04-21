import csv
from pathlib import Path
import argparse

from first_order_data.image_results import UNCERTAINTY_COLUMNS, load_json
from probly.conformal_prediction import (
    APSScore, LACScore, RAPSScore,
    SplitConformalClassifier,
    ClassConditionalClassifier,
    MondrianConformalClassifier,
    empirical_coverage, average_set_size,
)

import numpy as np


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


class CachedProbsModel:
    """Identity-style model: returns stored probabilities indexed by x.

    For post-hoc CP we already have probs on disk, so we just index into
    them. The API still requires a callable model because Score.calibration_nonconformity
    calls predict_probs(model, x_cal) internally.
    """
    def __init__(self, probs: np.ndarray) -> None:
        self._probs = np.asarray(probs, dtype=np.float64)

    def predict(self, x):
        # x is an integer index array; return the matching rows of probs.
        idx = np.asarray(x, dtype=int)
        return self._probs[idx]


def conformal_run(probs_all, targets_soft):
    y_all = targets_soft.argmax(axis=1)

    # 50/50 CAL/TEST split
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y_all))
    idx_cal, idx_test = perm[: len(perm) // 2], perm[len(perm) // 2 :]

    model = CachedProbsModel(probs_all)
    score = LACScore(model=model) # APSScore / RAPSScore also possible
    cp = SplitConformalClassifier(model=model, score=score)

    cp.calibrate(x_cal=idx_cal, y_cal=y_all[idx_cal], alpha=0.1) # target 90% coverage
    sets = cp.predict(x_test=idx_test, alpha=0.1) # bool (n_test, C)

    print("coverage:", empirical_coverage(sets, y_all[idx_test]))
    print("avg size:", average_set_size(sets))


def iter_dataset_rows(run_dir: Path):
    results = load_json(run_dir / "results.json")
    for dataset_result in results:
        for fold_result in dataset_result["folds"]:
            probs_all, targets_soft, uncertainty = load_prediction_csv(
                run_dir
                / dataset_result["dataset_name"]
                / dataset_result["encoder_name"]
                / fold_result["test_fold"]
                / "ensemble_predictions.csv"
            )
            print(f"\n{dataset_result['dataset_name']} / {fold_result['test_fold']}:")
            conformal_run(probs_all, targets_soft)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    iter_dataset_rows(args.run_dir)


if __name__ == "__main__":
    main()
