"""includes: class APSScore:

def calibration_non_conformity.

def predict_non_conformity.

def calculate_nonconformity_score (from aps/common.py)
"""

# clean version maybe
# probly/conformal_prediction/scores/aps.py
# -----------------------------------------
# AUS: aps/common.py::calculate_nonconformity_score (true-label-Version) [file:14]
# AUS: aps/torch.py::_compute_nonconformity, predict-Schleife [file:15]
# AUS: aps/flax.py::_compute_nonconformity, _create_prediction_sets [file:13]
# NEU: aps_scores_all_labels + APSScore (Score-Interface)

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .common import Score


def calculate_nonconformity_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """True-label APS Nonconformity-Scores (wie bisher in aps/common.py). [file:14]

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    labels : np.ndarray
        True labels of shape (n_samples).

    Returns:
    -------
    np.ndarray
        Non-conformity scores of shape (n_samples).
    """
    n_samples = probabilities.shape[0]
    scores = np.zeros(n_samples)

    for i in range(n_samples):
        probs = probabilities[i]
        sorted_items = sorted([(-probs[j], j) for j in range(len(probs))])
        # Get descending sorted probabilities
        sorted_indices = [idx for (_, idx) in sorted_items]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # find pos of true label in sorted order
        # will fail if labels[i] is out of bounds, but is expected
        true_label_pos = sorted_indices.index(labels[i])
        scores[i] = cumulative_probs[true_label_pos].item()

    return scores


# from aps/common.py (must be corrected according to alireza)
# changed to aps_scores_all_labels(probabilities)
def aps_scores_all_labels(
    probabilities: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """APS-Scores für alle Labels (n_instances x n_labels).

    Erweiterung von calculate_nonconformity_score auf alle Klassen.
    AUS: aps/common.py-Logik + Idee aus aps/flax._create_prediction_sets. [file:14][file:13]
    """
    n_samples, n_classes = probabilities.shape
    scores = np.zeros((n_samples, n_classes), dtype=float)

    for i in range(n_samples):
        probs = probabilities[i]
        sorted_items = sorted([(-probs[j], j) for j in range(n_classes)])
        sorted_indices = [idx for (_, idx) in sorted_items]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Score für jede Klasse = kumulative Wahrscheinlichkeit bis zu dieser Klasse
        for pos, cls in enumerate(sorted_indices):
            scores[i, cls] = cumulative_probs[pos]

    return scores


def create_aps_prediction_sets(
    probs: npt.NDArray[np.floating],
    threshold: float,
) -> npt.NDArray[np.bool_]:
    """Erzeuge APS-Prediction-Sets als 0/1-Matrix aus Wahrscheinlichkeiten.

    Kombiniert die Logik aus:
    - aps/torch.py::predict (Sortieren, kumulativ, <= threshold) [file:15]
    - aps/flax.py::_create_prediction_sets (immer min. ein Label) [file:13]
    """
    n_samples, n_classes = probs.shape
    prediction_sets = np.zeros((n_samples, n_classes), dtype=bool)

    for i in range(n_samples):
        sample_probs = probs[i]

        # Sort indices by probability (descending)
        sorted_indices = np.argsort(sample_probs)[::-1]
        sorted_probs = sample_probs[sorted_indices]

        cumulative = 0.0
        current_indices: list[int] = []

        for idx, prob in zip(sorted_indices, sorted_probs, strict=False):
            current_indices.append(int(idx))
            cumulative += float(prob)
            # klassische APS-Regel: cumulative > threshold -> stoppen
            if cumulative > threshold:
                break

        # Always include at least one class
        if not current_indices:
            current_indices = [int(sorted_indices[0])]

        prediction_sets[i, current_indices] = True

    return prediction_sets


class APSScore(Score):
    """APS Nonconformity-Score, backend-agnostisch (arbeitet mit numpy-Probas).

    model: Wrapper mit predict(x) -> np.ndarray der Form (n_samples, n_classes).
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> np.ndarray:
        """True-label-Scores für Kalibrierung.

        Nutzt aps_scores_all_labels, nimmt daraus die Scores für das wahre Label.
        Entspricht inhaltlich eurem alten _compute_nonconformity in Torch/Flax. [file:15][file:13]
        """
        probabilities = self.model.predict(x_cal)  # (n,k)
        all_scores = aps_scores_all_labels(probabilities)
        y_array = np.asarray(y_cal, dtype=int)
        n = len(y_array)
        return all_scores[np.arange(n), y_array]

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> np.ndarray:
        """Score-Matrix (n_instances x n_labels), wie vom Experten gefordert.

        In der einfachsten Version verwenden wir direkt aps_scores_all_labels.
        Alternativ könnt ihr hier auch schon eine 0/1-Matrix zurückgeben und
        die Mengenbildung in SplitConformalPredictor entsprechend anpassen.
        """
        probabilities = self.model.predict(x_test)
        return aps_scores_all_labels(probabilities)
