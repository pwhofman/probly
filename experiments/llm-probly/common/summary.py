"""Collect the results of all methods into one comparison table."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from common.evaluate import bootstrap_auroc, ece, evaluate, paired_bootstrap
from common.io import RESULTS_ROOT, load_scores
from common.registry import REGISTRY

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

COLUMNS = [
    "method",
    "status",
    "auroc",
    "ci_low",
    "ci_high",
    "vs_best",  # "best", "tied" (paired 95 % interval of the difference contains 0) or "worse"
    "d_auroc",  # AUROC minus the best method's AUROC
    "d_ci_low",
    "d_ci_high",
    "aurc",
    "prr",
    "ece",
    "n_failed",
    "seconds_per_question",
]

# Methods whose score is a probability of being correct (uncertainty = 1 - probability); only for these
# ECE is computed. Registered methods set probability=True in common/registry.py; this tuple adds the
# week-1 result names. A record can still override it with "probability": true.
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
    uncertainties = {}  # method -> uncertainty, for the paired comparison with the best method
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
        uncertainties[method] = record["uncertainty"]
        scores = evaluate(record["uncertainty"], correct)
        spec = REGISTRY.get(method)
        default = spec.probability if spec else method in PROBABILITY_METHODS
        is_probability = record.get("probability", default)
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
    add_paired_comparison(table, uncertainties, correct, n_boot=n_boot)
    return pd.DataFrame(table, columns=pd.Index(COLUMNS))


def add_paired_comparison(
    table: list[dict],
    uncertainties: dict[str, list[float]],
    correct: list[bool],
    n_boot: int = 1000,
) -> None:
    """Compare every evaluated method with the best one (highest AUROC) by a paired bootstrap (E4).

    Fills ``vs_best``, ``d_auroc``, ``d_ci_low`` and ``d_ci_high`` in place.
    ``tied`` means the data cannot tell the method apart from the best one: the
    95 % interval of its AUROC difference contains 0. This is the same test the
    selector (D1) uses to pick the cheapest method among the tied ones.
    """
    evaluated = [row for row in table if row["status"] == "ok"]
    if not evaluated:
        return
    best = max(evaluated, key=lambda row: row["auroc"])
    for row in evaluated:
        if row is best:
            row.update(vs_best="best", d_auroc=0.0, d_ci_low=0.0, d_ci_high=0.0)
            continue
        result = paired_bootstrap(uncertainties[row["method"]], uncertainties[best["method"]], correct, n_boot=n_boot)
        row.update(
            vs_best="tied" if result.verdict == "tie" else "worse" if result.verdict == "B" else "better",
            d_auroc=result.diff,
            d_ci_low=result.ci_low,
            d_ci_high=result.ci_high,
        )


def to_markdown(table: pd.DataFrame) -> str:
    """Write a DataFrame as a Markdown table (numbers with 3 decimals, NaN as an empty cell)."""

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.3f}"
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
    wide = long.pivot(index="method", columns="model", values=metric)[models]  # noqa: PD010 - one value per cell
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
    return pd.DataFrame(records, columns=pd.Index(SANITY_COLUMNS))
