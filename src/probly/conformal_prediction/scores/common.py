"""common structures for CP scores."""
# probly/conformal_prediction/scores/common.py
# -------------------------------------------
# AUS: aps/common.py::calculate_quantile [file:14]
# NEU: Score-Interface für APS, LAC, RAPS, ...

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class Score(Protocol):
    """Abstrakte Schnittstelle für Nonconformity-Scores.

    Jeder Score (APS, LAC, RAPS, ...) muss diese beiden Methoden implementieren.
    - calibration_nonconformity: wird in der Kalibrierungsphase aufgerufen.
    - predict_nonconformity: wird in der Testphase aufgerufen und soll
      eine Score-Matrix (n_instances x n_labels) liefern.
    """

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> np.ndarray: ...

    # Rückgabe: 1D-Array mit Scores pro Instanz (true-label-Scores)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> np.ndarray: ...

    # Rückgabe: 2D-Array (n_instances, n_labels) mit Scores für alle Labels


def calculate_quantile(scores: np.ndarray, alpha: float) -> float:
    """Berechne das (1 - alpha)-Quantil der Nonconformity-Scores.

    AUS: aps/common.py::calculate_quantile [file:14]

    Parameters
    ----------
    scores : np.ndarray
        Nonconformity-Scores der Kalibrierungsinstanzen.
    alpha : float
        Signifikanzniveau (target coverage = 1 - alpha).

    Returns:
    -------
    float
        (1 - alpha)-Quantil der Scores.
    """
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # ensure within [0, 1]
    return float(np.quantile(scores, q_level, method="lower"))
