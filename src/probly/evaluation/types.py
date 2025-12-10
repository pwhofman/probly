"""Data structures for OOD evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class OodEvaluationResult:
    """Standardized container for OOD evaluation results and curve data.

    Used to transfer data from the API calculation layer to the plotting layer.
    """

    # Scalar Metrics (for plot titles/legends)
    auroc: float
    aupr: float
    fpr95: float | None = None

    # Curve Vectors (required for plotting)
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
    precision: np.ndarray | None = None
    recall: np.ndarray | None = None

    # Raw Scores (optional, for histograms)
    id_scores: np.ndarray | None = None
    ood_scores: np.ndarray | None = None
