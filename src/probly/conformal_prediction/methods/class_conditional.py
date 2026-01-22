"""Class-Conditional Conformal Prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.conformal_prediction.methods.common import Predictor
    from probly.conformal_prediction.scores.common import ClassificationScore, RegressionScore

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.methods.common import (
    ConformalClassifier,
    ConformalRegressor,
    predict_probs,
)
from probly.conformal_prediction.methods.mondrian import (
    ClassFunc,
    GroupedConformalBase,
)
from probly.conformal_prediction.scores.lac.common import accretive_completion


class ClassConditionalClassifier(GroupedConformalBase, ConformalClassifier):
    """Class conditional conformal predictor for classification."""

    score: ClassificationScore

    def __init__(
        self,
        model: Predictor,
        score: ClassificationScore,
        class_func: ClassFunc,
        use_accretive: bool = False,
    ) -> None:
        """Create class conditional conformal predictor for classification."""
        # group_func is our class_func for setting thresholds per class
        super().__init__(model=model, group_func=class_func)
        self.score = score
        self.use_accretive = use_accretive

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return class-conditional prediction sets."""
        if not self.is_calibrated or not self.group_thresholds:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        scores = self.score.predict_nonconformity(x_test)
        if scores.ndim != 2:
            msg = "predict_nonconformity must return 2D-Matrix."
            raise ValueError(msg)

        scores_np = self.to_numpy(scores)
        n_labels = scores_np.shape[1]

        thresholds_array = np.full(n_labels, np.inf, dtype=float)

        for class_id, threshold in self.group_thresholds.items():
            if 0 <= class_id < n_labels:
                # ensure float for the array
                thresholds_array[int(class_id)] = float(threshold)

            thresholds = thresholds_array[None, :]  # (N, C)<=(1, C): broadcasting thresholds to all samples
        prediction_sets = scores_np <= thresholds

        if self.use_accretive:
            if probs is None:
                probs_raw: npt.NDArray[np.floating] = predict_probs(self.model, x_test)
                probs_np = self.to_numpy(probs_raw)
            else:
                probs_np = probs
            prediction_sets = accretive_completion(prediction_sets, probs_np)

        return prediction_sets.astype(bool)


class ClassConditionalRegressor(GroupedConformalBase, ConformalRegressor):
    """Class-conditional conformal predictor for regression."""

    score: RegressionScore

    def __init__(
        self,
        model: Predictor,
        score: RegressionScore,
        class_func: ClassFunc,
    ) -> None:
        """Create class conditional conformal predictor for regression."""
        # directly use GroupedConformalBase to pass class_func correctly
        super().__init__(model=model, group_func=class_func)
        self.score = score

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
    ) -> npt.NDArray[np.floating]:
        """Return prediction intervals based on class groups."""
        if not self.is_calibrated:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        y_hat = self.model(x_test)
        y_hat_np = self.to_numpy(y_hat)

        # group_func is class_func, might predict the class of x_test
        region_ids = self.group_func(x_test, None)
        region_ids_np = np.asarray(region_ids, dtype=int)

        if self.is_asymmetric:
            if y_hat_np.ndim != 2 or y_hat_np.shape[1] != 2:
                msg = "Asymmetric intervals expect model output shape (N, 2)."
                raise ValueError(msg)

            if not self.group_thresholds_lower or not self.group_thresholds_upper:
                msg = "Asymmetric thresholds not calibrated."
                raise RuntimeError(msg)

            # reuse helper from Base class
            threshold_lower, threshold_upper = self._get_thresholds_per_sample_asym(region_ids_np)

            lower = y_hat_np[:, 0] - threshold_lower
            upper = y_hat_np[:, 1] + threshold_upper
        else:
            y_hat_flat = y_hat_np.flatten()

            if not self.group_thresholds:
                msg = "Standard threshold is not calibrated."
                raise RuntimeError(msg)

            # reuse helper from Base class
            thresholds = self._get_thresholds_per_sample(region_ids_np)
            lower = y_hat_flat - thresholds
            upper = y_hat_flat + thresholds

        return cast("npt.NDArray[np.floating]", np.stack([lower, upper], axis=1))
