"""Split conformal prediction methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.methods.common import ConformalPredictor, PredictiveModel
from probly.conformal_prediction.scores.common import Score
from probly.conformal_prediction.scores.lac import accretive_completion
from probly.conformal_prediction.utils import calculate_quantile


class SplitConformalPredictor(ConformalPredictor):
    """Generic split conformal predictor for classification."""

    def __init__(
        self,
        model: PredictiveModel,
        score: Score,
        use_accretive: bool = False,
    ) -> None:
        """Create a split conformal predictor.

        Args:
            model: Wrapper that provides predict(x) -> np.ndarray of probabilities.
            score: Score object implementing the nonconformity computations.
            use_accretive: If True, apply accretive completion to handle empty sets.
        """
        super().__init__(model=model)
        self.score = score
        self.use_accretive = use_accretive

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate the predictor on a calibration dataset.

        Computes nonconformity scores on the calibration data via the
        score object, then stores the (1 - alpha)-quantile as threshold.
        """
        # nonconformity score from object
        self.nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)

        # calculate quantile threshold
        self.threshold = calculate_quantile(self.nonconformity_scores, alpha)

        self.is_calibrated = True
        return self.threshold

    def predict(self, x_test: Sequence[Any]) -> npt.NDArray[np.bool_]:
        """Return prediction sets as a (n_instances, n_labels) 0/1-matrix.

        For APS: the score matrix is based on cumulative probabilities.
        For LAC: scores typically follow s(x, y) = 1 - p(y | x) for all labels.
        Labels with score <= threshold are included in the prediction set.
        """
        if not self.is_calibrated or self.threshold is None:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        scores = self.score.predict_nonconformity(x_test)  # shape: matrix (n_instances, n_labels)

        if scores.ndim != 2:
            msg = "predict_nonconformity must return 2D-Matrix (n_instances, n_labels)."
            raise ValueError(msg)

        # sets defined: label included when score <= threshold
        prediction_sets = scores <= self.threshold  # bool-Array (n_instances, n_labels)

        # accretive completion for empty sets
        if self.use_accretive:
            # for LAC: scores = 1 - p(y|x), also p(y|x) = 1 - scores
            probs = 1.0 - scores
            prediction_sets = accretive_completion(prediction_sets, probs)

        return prediction_sets
