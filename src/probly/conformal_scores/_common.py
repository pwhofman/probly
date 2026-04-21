"""Contract definitions for nonconformity scores."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class NonConformityScore[In, Out](Protocol):
    """Protocol for nonconformity-score callables."""

    def __call__(self, y_pred: In, y_true: In | None = None) -> Out:
        """Obtain the nonconformity score for calibration or prediction."""
        ...
