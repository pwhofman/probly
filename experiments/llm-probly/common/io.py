"""Save, load and validate uncertainty scores according to CONTRACT.md."""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

REQUIRED_KEYS = (
    "method",
    "model",
    "answer",
    "ids",
    "uncertainty",
    "failed",
    "seconds_per_question",
)


def validate_scores(record: dict, expected_ids: list[int] | None = None) -> list[str]:
    """Return a list of problems with a score record (empty list means valid)."""
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        return [f"missing keys: {missing}"]

    n = len(record["ids"])
    if len(record["uncertainty"]) != n or len(record["failed"]) != n:
        return ["ids, uncertainty and failed must have the same length"]

    problems = []
    if record["answer"] != "greedy":
        problems.append(f"answer must be 'greedy', got {record['answer']!r}")

    for i, (u, failed) in enumerate(zip(record["uncertainty"], record["failed"])):
        if not isinstance(u, (int, float)) or math.isnan(u) or math.isinf(u):
            problems.append(f"uncertainty[{i}] is not a finite number: {u!r}")
        elif failed and u != 1.0:
            problems.append(f"uncertainty[{i}] must be 1.0 when failed is true, got {u}")

    if expected_ids is not None and list(record["ids"]) != list(expected_ids):
        problems.append("ids do not match the expected ids (same values, same order)")

    return problems


def save_scores(record: dict, run: str = "qwen0.5b", expected_ids: list[int] | None = None) -> Path:
    """Validate a score record and save it as results/<run>/<method>.json."""
    problems = validate_scores(record, expected_ids)
    if problems:
        raise ValueError("invalid scores: " + "; ".join(problems))
    path = RESULTS_ROOT / run / f"{record['method']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


def load_scores(path: str | Path, expected_ids: list[int] | None = None) -> dict:
    """Load a score file and check it against the contract."""
    record = json.loads(Path(path).read_text())
    problems = validate_scores(record, expected_ids)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return record
