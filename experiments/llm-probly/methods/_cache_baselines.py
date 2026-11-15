"""Shared helper for the baselines that only read the cache (BIG).

Not a plugin (the runner skips files that start with "_").
"""

from __future__ import annotations

import math


def record(name: str, rows: list[dict], values: list[float | None], seconds: float = 0.0) -> dict:
    """Build a score record from one value per row (higher = more uncertain).

    A row without a value (the greedy answer has no scored tokens) is marked
    failed and gets uncertainty 1.0, as CONTRACT.md rule 3 requires.
    """
    failed = [v is None or not math.isfinite(v) for v in values]
    return {
        "method": name,
        "model": rows[0]["model"],
        "answer": "greedy",
        "ids": [row["id"] for row in rows],
        "uncertainty": [1.0 if bad else float(v) for v, bad in zip(values, failed, strict=True)],
        "failed": failed,
        "seconds_per_question": seconds,
    }
