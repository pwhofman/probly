"""Evaluation namespace: re-exports the predicted-set metrics from :mod:`probly.metrics`.

The implementations live in :mod:`probly.metrics` (the canonical home for all
scoring functions). They are re-exported here so that the representation-level
workflow code that lives in :mod:`probly.evaluation` (active learning, OOD
detection, selective prediction) can import them from the same namespace.
"""

from __future__ import annotations

from probly.metrics import coverage, efficiency

__all__ = [
    "coverage",
    "efficiency",
]
