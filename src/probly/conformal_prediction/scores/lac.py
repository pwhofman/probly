"""includes: class LACScore:

def calibration_non_conformity.

def predict_non_conformity.
"""

# clean version maybe

# probly/conformal_prediction/scores/lac.py
# ----------------------------------------
# AUS: lac/common.py::calculate_non_conformity_score, calculate_local_weights,
#      calculate_weighted_quantile, accretive_completion, LAC.predict [file:16]
# AUS: lac/torch.py::predict (Threshold-Logik, Accretive) [file:12]
# AUS: lac/flax.py::predict (Threshold-Logik, Accretive) [file:11]
# NEU: LACScore + all-label-Scores

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .common import Score

# --------- Nonconformity / Quantile / Weights --------- #


def calculate_non_conformity_score_true_label(
    probas: npt.NDArray[np.floating],
    y_indices: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:
    """Compute LAC Non-Conformity Scores für das wahre Label.

    AUS: lac/common.py::calculate_non_conformity_score [file:16]

    s(x, y_true) = 1 - p(y_true | x)
    """
    n_samples = len(y_indices)
    true_class_probas = probas[np.arange(n_samples), y_indices]
    scores = 1.0 - true_class_probas
    return scores


def calculate_non_conformity_scores_all_labels(
    probas: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """LAC Nonconformity-Scores für alle Labels.

    Erweiterung von calculate_non_conformity_score_true_label auf alle Klassen.
    s(x, y) = 1 - p(y | x) für jede Klasse y.
    """
    return 1.0 - probas  # shape: (n_samples, n_classes)


def calculate_local_weights(
    x: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Berechnet lokale Gewichte.

    AUS: lac/common.py::calculate_local_weights [file:16]

    Für Standard-LAC Split: uniforme Gewichte.
    """
    n_samples = x.shape[0]
    return np.ones(n_samples, dtype=float)


def calculate_weighted_quantile(
    values: npt.NDArray[np.floating],
    quantile: float,
    sample_weight: npt.NDArray[np.floating] | None = None,
) -> float:
    """Berechnet einen gewichteten Quantil-Schätzer mittels numpy.

    AUS: lac/common.py::calculate_weighted_quantile [file:16]
    """
    if sample_weight is None:
        return float(np.quantile(values, quantile, method="higher"))

    values = np.array(values)
    sample_weight = np.array(sample_weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)

    return float(np.interp(quantile, weighted_quantiles, values))


# --------- Accretive Completion --------- #


def accretive_completion(
    prediction_sets: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    """Eliminiert leere Vorhersagesets (Null Regions) via Accretive Completion.

    AUS: lac/common.py::accretive_completion [file:16]

    Wenn ein Set leer ist, wird die Klasse mit der höchsten Wahrscheinlichkeit
    (scores) hinzugefügt.
    """
    completed_sets = prediction_sets.copy()

    set_sizes = np.sum(completed_sets, axis=1)
    empty_rows_mask = set_sizes == 0

    if not np.any(empty_rows_mask):
        return completed_sets

    # Für leere Zeilen: Index der Klasse mit höchster Wahrscheinlichkeit
    best_class_indices = np.argmax(scores[empty_rows_mask], axis=1)
    row_indices = np.where(empty_rows_mask)[0]

    completed_sets[row_indices, best_class_indices] = True
    return completed_sets


# --------- Helper: Sets aus Probas --------- #


def create_lac_prediction_sets(
    probas: npt.NDArray[np.floating],
    threshold: float,
    use_accretive: bool = True,
) -> npt.NDArray[np.bool_]:
    """Erzeuge LAC-Prediction-Sets als 0/1-Matrix aus Wahrscheinlichkeiten.

    Kombiniert die Logik aus:
    - lac/common.py::LAC.predict (Threshold + accretive_completion) [file:16]
    - lac/torch.py::predict (gleiche Idee in Torch) [file:12]
    - lac/flax.py::predict (gleiche Idee in Flax) [file:11]
    """
    # Score <= t <=> 1 - p <= t <=> p >= 1 - t
    prob_threshold = 1.0 - threshold
    prediction_sets = probas >= prob_threshold  # bool (n_samples, n_classes)

    if use_accretive:
        prediction_sets = accretive_completion(prediction_sets, probas)

    return prediction_sets


# --------- Score-Klasse --------- #


class LACScore(Score):
    """LAC Nonconformity-Score, backend-agnostisch (arbeitet mit numpy-Probas).

    - model: Wrapper mit predict(x) -> np.ndarray (n_samples, n_classes),
      z.B. TorchModelWrapper oder FlaxWrapper.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> np.ndarray:
        """True-label-Scores für Kalibrierung.

        Entspricht inhaltlich _compute_nonconformity aus LAC.common/Flax/Torch. [file:16][file:11][file:12]
        """
        probas = self.model.predict(x_cal)  # (n,k)
        y_indices = np.asarray(y_cal, dtype=int)
        return calculate_non_conformity_score_true_label(probas, y_indices)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> np.ndarray:
        """Score-Matrix (n_instances x n_labels) für LAC.

        Wie im Expertenfeedback: Scores für alle Labels.
        s(x,y) = 1 - p(y|x).
        """
        probas = self.model.predict(x_test)  # (n,k)
        return calculate_non_conformity_scores_all_labels(probas)
