# clean version maybe
# probly/conformal_prediction/common.py
# ------------------------------------
# Zentrale Split-Conformal-Implementierung + gemeinsame Basis
#
# NUTZT:
#   - Score-Interface + calculate_quantile aus scores/common.py [file:14]
#   - accretive_completion aus scores/lac.py (optional) [file:16]
#
# VERLAGERT:
#   - calibrate/predict-Logik aus APSPredictor (aps/torch.py, aps/flax.py) [file:15][file:13]
#   - LAC.predict / LAC.calibrate-Idee aus lac/common.py, lac/torch.py, lac/flax.py [file:16][file:11][file:12]

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from probly.conformal_prediction.scores.common import Score, calculate_quantile
from probly.conformal_prediction.scores.lac import accretive_completion


class ConformalPredictor:
    """Einfache Basisklasse für Conformal-Predictoren.

    Falls ihr bereits eine eigene ConformalPredictor-Basisklasse habt,
    könnt ihr diese hier anpassen oder ersetzen.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.is_calibrated: bool = False
        self.threshold: float | None = None
        self.nonconformity_scores: np.ndarray | None = None

    def __str__(self) -> str:
        """String-Representation, angelehnt an APS/Flax.__str__. [file:13][file:15]"""
        model_name = self.model.__class__.__name__
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"{self.__class__.__name__}(model={model_name}, status={status})"


class SplitConformalPredictor(ConformalPredictor):
    """Generischer Split-Conformal-Predictor für Klassifikation.

    - Nutzt eine Score-Klasse (APSScore, LACScore, ...) über das Score-Interface.
    - Kapselt den gemeinsamen Workflow:
        * calibration_nonconformity -> Quantil -> threshold
        * predict_nonconformity    -> Mengenbildung (0/1-Matrix)
    """

    def __init__(
        self,
        model: Any,
        score: Score,
        use_accretive: bool = False,
    ) -> None:
        """Initialisiere SplitConformalPredictor.

        Args:
            model: Backend-Wrapper (z.B. TorchModelWrapper, FlaxWrapper),
                   der predict(x) -> np.ndarray Probas liefert.
            score: Score-Objekt (z.B. APSScore, LACScore).
            use_accretive: Wenn True, wird Accretive Completion angewendet
                           (typisch für LAC-artige Scores).
        """
        super().__init__(model=model)
        self.score = score
        self.use_accretive = use_accretive

    # ----------------- Kalibrierung ----------------- #

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Kalibriere den Prädiktor mit Kalibrierungsdaten.

        Entspricht der Logik aus APSPredictor.calibrate und LAC.calibrate,
        nur dass die Nonconformity-Berechnung an die Score-Klasse ausgelagert ist.
        [file:14][file:15][file:16]
        """
        # 1) Nonconformity-Scores vom Score-Objekt
        self.nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)

        # 2) Quantil (threshold) berechnen
        self.threshold = calculate_quantile(self.nonconformity_scores, alpha)

        # 3) Status setzen
        self.is_calibrated = True
        return self.threshold

    # ----------------- Vorhersage ----------------- #

    def predict(self, x_test: Sequence[Any]) -> np.ndarray:
        """Erzeuge Vorhersagesets als 0/1-Matrix (n_instances x n_labels).

        - Für APS: Score-Matrix basiert z.B. auf kumulativen Wahrscheinlichkeiten.
        - Für LAC: Score-Matrix typischerweise s(x,y) = 1 - p(y|x) (all labels).
        - Die Regel "Score <= threshold" definiert die Menge.
        Optional kann Accretive Completion angewandt werden (LAC).
        [file:15][file:16]
        """
        if not self.is_calibrated or self.threshold is None:
            raise RuntimeError("Predictor must be calibrated before predict().")

        # 1) Score-Matrix für alle Labels beziehen
        scores = self.score.predict_nonconformity(x_test)  # shape: (n_instances, n_labels)

        if scores.ndim != 2:
            msg = "predict_nonconformity muss eine 2D-Matrix (n_instances, n_labels) zurückgeben."
            raise ValueError(msg)

        # 2) Sets definieren: Label gehört zur Menge, wenn Score <= threshold
        prediction_sets = scores <= self.threshold  # bool-Array (n_instances, n_labels)

        # 3) Optional: Accretive Completion für leere Sets (typisch für LAC)
        if self.use_accretive:
            # Für LAC gilt: scores = 1 - p(y|x), also p(y|x) = 1 - scores
            probas = 1.0 - scores
            prediction_sets = accretive_completion(prediction_sets, probas)

        return prediction_sets
