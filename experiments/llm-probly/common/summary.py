"""Collect the results of all methods into one comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.evaluate import bootstrap_auroc, evaluate
from common.io import RESULTS_ROOT, load_scores

EXPECTED_METHODS = (
    "likelihood_mean",
    "likelihood_sum",
    "p_true_zeroshot_norm",
    "p_true_brainstorm",
    "p_true_empirical",
    "verbalized_2s",
    "semantic_entropy",
    "discrete_se",
)

COLUMNS = ["method", "status", "auroc", "ci_low", "ci_high", "aurc", "n_failed", "seconds_per_question"]

SANITY_ROOT = RESULTS_ROOT / "sanity"
SANITY_COLUMNS = ["method", "model", "auroc", "gap", "n_distinct", "verdict"]


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


def sanity_verdict(auroc: float) -> str:
    """Interpret a sanity test AUROC with the thresholds of the plan."""
    if auroc >= 0.8:
        return "works"
    if auroc >= 0.6:
        return "weak signal"
    return "random"


def sanity_table(folder: Path = SANITY_ROOT) -> pd.DataFrame:
    """Collect all sanity test results (``results/sanity/*.json``) into one table.

    One row per method and model. ``auroc`` is computed with the correct answer as
    the positive class and the confidence as the score (higher = more confident).
    """
    records = []
    for path in sorted(Path(folder).glob("*.json")):
        data = json.loads(path.read_text())
        if not {"method", "model", "auroc", "gap", "n_distinct"} <= data.keys():
            continue
        records.append(
            {
                "method": data["method"],
                "model": data["model"].split("/")[-1],
                "auroc": data["auroc"],
                "gap": data["gap"],
                "n_distinct": data["n_distinct"],
                "verdict": sanity_verdict(data["auroc"]),
            },
        )
    return pd.DataFrame(records, columns=SANITY_COLUMNS)
