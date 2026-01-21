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
from probly.conformal_prediction.scores.common import calculate_quantile
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
        super().__init__(model=model)
        self.score = score
        self.use_accretive = use_accretive
        self.group_func = class_func

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
        probs: Any = None,  # noqa: ANN401
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

        # get class ids for test samples
        thresholds_array = np.full(n_labels, np.inf, dtype=float)
        for class_id, threshold in self.group_thresholds.items():
            thresholds_array[int(class_id)] = threshold

        thresholds = thresholds_array[None, :]
        prediction_sets = scores_np <= thresholds

        if self.use_accretive:
            if probs is None:
                probs = predict_probs(self.model, x_test)
            probs_np = self.to_numpy(probs)
            prediction_sets = accretive_completion(prediction_sets, probs_np)

        return prediction_sets


class ClassConditionalRegressor(GroupedConformalBase, ConformalRegressor):
    """Class-conditional conformal predictor for regression."""

    score: RegressionScore
    group_thresholds_lower: dict[int, float]
    group_thresholds_upper: dict[int, float]
    is_cqr: bool

    def __init__(
        self,
        model: Predictor,
        score: RegressionScore,
        class_func: ClassFunc,
    ) -> None:
        """Create class conditional conformal predictor for regression."""
        super().__init__(model=model)
        self.score = score
        self.group_func = class_func
        self.group_thresholds_lower = {}
        self.group_thresholds_upper = {}
        self.is_asymmetric = False

    def get_thresholds_per_sample(self, class_ids_np: npt.NDArray[np.int_]) -> npt.NDArray[np.floating]:
        """Assign symmetric threshold per sample."""
        n_samples = class_ids_np.shape[0]
        thresholds = np.empty(n_samples, dtype=float)
        max_threshold = max(self.group_thresholds.values()) if self.group_thresholds else np.inf

        # get class ids for test samples
        for i, class_id in enumerate(class_ids_np):
            thresholds[i] = self.group_thresholds.get(int(class_id), max_threshold)
        return thresholds

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate thresholds per class."""
        nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)
        scores_np = self.to_numpy(nonconformity_scores)
        class_ids = self.group_func(x_cal, y_cal)
        class_ids_np = np.asarray(class_ids, dtype=int)

        if class_ids_np.shape[0] != scores_np.shape[0]:
            msg = "Class ids and scores must have same length."
            raise ValueError(msg)

        # determine if symmetric or asymmetric thresholds
        if scores_np.ndim == 1 or (scores_np.ndim == 2 and scores_np.shape[1] == 1):
            # standard: one threshold per class (symmetric)
            self.is_asymmetric = False
            scores_flat = scores_np.flatten()
            self.group_thresholds = {}
            unique_classes = np.unique(class_ids_np)

            # get class ids for calibration samples
            for class_id in unique_classes:
                class_mask = class_ids_np == class_id
                scores_in_class = scores_flat[class_mask]

                if scores_in_class.size > 0:
                    self.group_thresholds[int(class_id)] = calculate_quantile(scores_in_class, alpha)

        elif scores_np.ndim == 2 and scores_np.shape[1] == 2:
            # asymmetric: two thresholds per class (CQR)
            self.is_asymmetric = True
            self.group_thresholds_lower = {}
            self.group_thresholds_upper = {}

            unique_classes = np.unique(class_ids_np)
            alpha_lower = alpha / 2
            alpha_upper = 1 - alpha / 2

            # get class ids for calibration samples
            for class_id in unique_classes:
                class_mask = class_ids_np == class_id
                scores_lower = scores_np[class_mask, 0]
                scores_upper = scores_np[class_mask, 1]

                if scores_lower.size > 0:
                    self.group_thresholds_lower[int(class_id)] = calculate_quantile(scores_lower, alpha_lower)
                    self.group_thresholds_upper[int(class_id)] = calculate_quantile(scores_upper, alpha_upper)
        else:
            msg = f"Score shape {scores_np.shape} not supported."
            raise ValueError(msg)

        self.is_calibrated = True
        return alpha

    def get_thresholds_per_sample_asym(
        self,
        class_ids_np: npt.NDArray[np.int_],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Assign asymmetric thresholds (lower/upper) per sample."""
        n_samples = class_ids_np.shape[0]
        threshold_lower = np.empty(n_samples, dtype=float)
        threshold_upper = np.empty(n_samples, dtype=float)

        max_lower = max(self.group_thresholds_lower.values()) if self.group_thresholds_lower else np.inf
        max_upper = max(self.group_thresholds_upper.values()) if self.group_thresholds_upper else np.inf

        # get class ids for test samples
        for i, class_id in enumerate(class_ids_np):
            class_id_int = int(class_id)
            threshold_lower[i] = self.group_thresholds_lower.get(class_id_int, max_lower)
            threshold_upper[i] = self.group_thresholds_upper.get(class_id_int, max_upper)

        return threshold_lower, threshold_upper

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
    ) -> npt.NDArray[np.floating]:
        """Return class-conditional intervals."""
        if not self.is_calibrated:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        y_hat = self.model(x_test)
        y_hat_np = self.to_numpy(y_hat)

        if self.is_asymmetric:
            if y_hat_np.ndim != 2 or y_hat_np.shape[1] != 2:
                msg = "Asymmetric intervals expect model output shape (N, 2)."
                raise ValueError(msg)

            n_test = y_hat_np.shape[0]
            class_ids = self.group_func(x_test, None)
            class_ids_np = np.asarray(class_ids, dtype=int)

            if class_ids_np.shape[0] != n_test:
                msg = "Class ids must match test size."
                raise ValueError(msg)

            if not self.group_thresholds_lower or not self.group_thresholds_upper:
                msg = "Asymmetric thresholds not calibrated."
                raise RuntimeError(msg)

            threshold_lower, threshold_upper = self.get_thresholds_per_sample_asym(class_ids_np)

            lower = y_hat_np[:, 0] - threshold_lower
            upper = y_hat_np[:, 1] + threshold_upper
        else:
            y_hat_flat = y_hat_np.flatten()
            n_test = y_hat_flat.shape[0]
            class_ids = self.group_func(x_test, None)
            class_ids_np = np.asarray(class_ids, dtype=int)

            if class_ids_np.shape[0] != n_test:
                msg = "Class ids must match test size."
                raise ValueError(msg)

            if not self.group_thresholds:
                msg = "Standard thresholds not calibrated."
                raise RuntimeError(msg)

            thresholds = self.get_thresholds_per_sample(class_ids_np)
            lower = y_hat_flat - thresholds
            upper = y_hat_flat + thresholds

        return cast("npt.NDArray[np.floating]", np.stack([lower, upper], axis=1))
