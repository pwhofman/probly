"""Collect the results of all methods into one comparison table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.evaluate import bootstrap_auroc, evaluate
from common.io import RESULTS_ROOT, load_scores

EXPECTED_METHODS = (
    "p_true_zeroshot_norm",
    "p_true_brainstorm",
    "p_true_empirical",
    "verbalized_2s",
    "semantic_entropy",
    "discrete_se",
)

COLUMNS = ["method", "status", "auroc", "ci_low", "ci_high", "aurc", "n_failed", "seconds_per_question"]


def summary_table(
    rows: list[dict],
    run: str = "qwen0.5b",
    methods: tuple[str, ...] = EXPECTED_METHODS,
    n_boot: int = 1000,
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """Evaluate every method file in ``results_root/run`` and return one row per method.

    ``rows`` are the shared data rows (B1) with the keys ``id`` and ``correct``.
    Methods without a valid file get a row with ``status`` set to ``pending``.
    """
    ids = [row["id"] for row in rows]
    correct = [row["correct"] for row in rows]

    folder = Path(results_root) / run
    found = sorted(path.stem for path in folder.glob("*.json")) if folder.exists() else []
    all_methods = [*methods, *(name for name in found if name not in methods)]

    table = []
    for method in all_methods:
        path = folder / f"{method}.json"
        if not path.exists():
            table.append({"method": method, "status": "pending: no file yet"})
            continue
        try:
            record = load_scores(path, expected_ids=ids)
        except ValueError as error:
            table.append({"method": method, "status": f"pending: {error}"})
            continue
        scores = evaluate(record["uncertainty"], correct)
        low, high = bootstrap_auroc(record["uncertainty"], correct, n_boot=n_boot)
        table.append(
            {
                "method": method,
                "status": "ok",
                "auroc": scores["auroc"],
                "ci_low": low,
                "ci_high": high,
                "aurc": scores["aurc"],
                "n_failed": int(sum(record["failed"])),
                "seconds_per_question": record["seconds_per_question"],
            },
        )
    return pd.DataFrame(table, columns=COLUMNS)
