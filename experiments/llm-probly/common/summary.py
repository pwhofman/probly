"""Collect the results of all methods into one comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.evaluate import bootstrap_auroc, ece, evaluate
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

COLUMNS = ["method", "status", "auroc", "ci_low", "ci_high", "aurc", "prr", "ece", "n_failed", "seconds_per_question"]

# Methods whose score is a probability of being correct (uncertainty = 1 - probability).
# Only for these ECE is computed. New plugins can instead put "probability": true in their record.
PROBABILITY_METHODS = (
    "p_true_zeroshot",
    "p_true_zeroshot_norm",
    "p_true_brainstorm",
    "p_true_empirical",
    "verbalized_2s",
)

# Model sizes from small to large; the comparison table uses this column order.
MODEL_ORDER = ("smollm135m", "smollm2", "qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b")

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
        is_probability = record.get("probability", method in PROBABILITY_METHODS)
        calibration = ece(record["uncertainty"], correct) if is_probability else float("nan")
        low, high = bootstrap_auroc(record["uncertainty"], correct, n_boot=n_boot)
        table.append(
            {
                "method": method,
                "status": "ok",
                "auroc": scores["auroc"],
                "ci_low": low,
                "ci_high": high,
                "aurc": scores["aurc"],
                "prr": scores["prr"],
                "ece": calibration,
                "n_failed": int(sum(record["failed"])),
                "seconds_per_question": record["seconds_per_question"],
            },
        )
    return pd.DataFrame(table, columns=COLUMNS)


def to_markdown(table: pd.DataFrame) -> str:
    """Write a DataFrame as a Markdown table (numbers with 3 decimals, NaN as an empty cell)."""

    def cell(value) -> str:
        if isinstance(value, float):
            return "" if value != value else f"{value:.3f}"  # value != value is True only for NaN
        return str(value)

    lines = ["| " + " | ".join(str(c) for c in table.columns) + " |", "|" + "---|" * len(table.columns)]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def find_models(run_name: str, results_root: Path = RESULTS_ROOT) -> list[str]:
    """Models that have a results folder ``results/<model>/<run_name>/``, small to large."""
    found = [p.parent.name for p in Path(results_root).glob(f"*/{run_name}") if p.is_dir()]
    known = [m for m in MODEL_ORDER if m in found]
    return known + sorted(m for m in found if m not in MODEL_ORDER)


def compare_models(
    rows_by_model: dict[str, list[dict]],
    run_name: str,
    n_boot: int = 1000,
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """One long table: every method of every model, with a ``model`` and an ``accuracy`` column.

    ``rows_by_model`` maps a model to its labelled cache rows, ``run_name`` is
    the folder inside each model, e.g. ``"triviaqa_200"``.
    """
    parts = []
    for model, rows in rows_by_model.items():
        table = summary_table(rows, run=f"{model}/{run_name}", methods=(), n_boot=n_boot, results_root=results_root)
        table = table[table["status"] == "ok"].drop(columns="status")
        table.insert(0, "model", model)
        table.insert(2, "accuracy", sum(r["correct"] for r in rows) / len(rows))
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def method_by_model(long: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Turn the long table into methods x models for one metric (e.g. "auroc")."""
    models = list(dict.fromkeys(long["model"]))  # keep the small-to-large order
    wide = long.pivot(index="method", columns="model", values=metric)[models]
    wide = wide.dropna(how="all")  # e.g. ECE: drop methods that are not probabilities
    return wide.reset_index().rename_axis(columns=None)


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
