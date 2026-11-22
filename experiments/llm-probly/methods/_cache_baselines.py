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
    clean = [v if v is not None and math.isfinite(v) else None for v in values]
    failed = [v is None for v in clean]
    return {
        "method": name,
        "model": rows[0]["model"],
        "answer": "greedy",
        "ids": [row["id"] for row in rows],
        "uncertainty": [1.0 if v is None else float(v) for v in clean],
        "failed": failed,
        "seconds_per_question": seconds,
    }
