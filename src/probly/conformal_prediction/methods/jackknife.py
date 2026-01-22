"""Jackknife Conformal Prediction Methods."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.conformal_prediction.methods.common import Predictor

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold, LeaveOneOut
import torch
from torch import Tensor

from probly.conformal_prediction.methods.common import (
    ConformalClassifier,
    ConformalPredictor,
    ConformalRegressor,
    predict_probs,
)
from probly.conformal_prediction.scores.lac.common import accretive_completion
from probly.conformal_prediction.utils.quantile import calculate_quantile

# --- Type Definitions ---
ScoreFunc = Callable[[npt.NDArray, npt.NDArray], npt.NDArray[np.floating]]
IntervalFunc = Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]


class JackknifeCVBase(ConformalPredictor):
    """Base class for resampling-based conformal prediction (Jackknife+ / CV+)."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int | Any | None = None,  # noqa: ANN401
        random_state: int | None = None,
    ) -> None:
        """Initialize the JackknifeCVBase."""
        super().__init__(model=None)  # type: ignore[arg-type]
        self.model_factory = model_factory
        self.cv = cv
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.fitted_models: list[Predictor] = []
        self.fold_assignments: npt.NDArray[np.int_] | None = None
        self.nonconformity_scores: npt.NDArray[np.floating] | None = None
        self.n_folds_actual_: int = 0

    @staticmethod
    def to_numpy(x: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Convert input to numpy array of floats."""
        if torch is not None and isinstance(x, Tensor):
            return cast("npt.NDArray[np.floating]", x.detach().cpu().numpy())
        return np.asarray(x, dtype=float)

    def _get_cv_splitter(self, n_samples: int) -> Any:  # noqa: ANN401
        """Return the cross-validation splitter."""
        if self.cv is None:
            return LeaveOneOut()

        if isinstance(self.cv, int):
            if self.cv == -1 or self.cv >= n_samples:
                return LeaveOneOut()

            if self.cv < 2:
                msg = f"cv must be >= 2 (or -1 for LeaveOneOut), got {self.cv}"
                raise ValueError(msg)
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        if hasattr(self.cv, "split"):
            return self.cv
        msg = f"Unknown cv type: {type(self.cv)}"
        raise ValueError(msg)

    def create_fold_assignments(self, x: npt.NDArray, y: npt.NDArray) -> npt.NDArray[np.int_]:
        """Create fold assignments for each sample based on the CV strategy."""
        n_samples = len(x)
        splitter = self._get_cv_splitter(n_samples)
        assignments = np.full(n_samples, -1, dtype=int)
        fold_count = 0

        for fold_idx, (_, test_idx) in enumerate(splitter.split(x, y)):
            assignments[test_idx] = fold_idx
            fold_count = fold_idx + 1

        if np.any(assignments == -1):
            msg = "CV strategy must assign every sample to a test fold."
            raise ValueError(msg)
        self.n_folds_actual_ = fold_count
        return assignments

    @abstractmethod
    def predict_fold(self, model: Predictor, x: npt.NDArray) -> npt.NDArray[np.floating]:
        """Predict using the given model on the provided data."""

    @abstractmethod
    def compute_scores(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores based on true and predicted values."""

    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], alpha: float) -> float:
        """Calibrate the Jackknife+ / CV+ predictor."""
        x_np = np.asarray(x_cal)
        y_np = np.asarray(y_cal)
        n_samples = len(x_np)

        self.fold_assignments = self.create_fold_assignments(x_np, y_np)
        self.fitted_models = []

        oof_predictions_list = []

        for fold in range(self.n_folds_actual_):
            val_mask = self.fold_assignments == fold
            train_mask = ~val_mask

            if not np.any(val_mask):
                continue

            model = self.model_factory()
            if hasattr(model, "fit"):
                model.fit(x_np[train_mask], y_np[train_mask])
            self.fitted_models.append(model)

            prediction = self.predict_fold(model, x_np[val_mask])
            oof_predictions_list.append((np.where(val_mask)[0], prediction))

        first_prediction = oof_predictions_list[0][1]
        if first_prediction.ndim == 1:
            oof_predictions = np.zeros(n_samples)
        else:
            oof_predictions = np.zeros((n_samples, first_prediction.shape[1]))

        for indices, prediction in oof_predictions_list:
            oof_predictions[indices] = prediction

        self.nonconformity_scores = self.compute_scores(y_np, oof_predictions)
        scores_np = self.to_numpy(self.nonconformity_scores)

        self.threshold = calculate_quantile(scores_np, alpha)
        self.is_calibrated = True
        return self.threshold

    def get_aligned_predictions(self, x_test: npt.NDArray) -> npt.NDArray[np.floating]:
        """Get predictions from each fold model aligned to original data order."""
        predictions = []
        for model in self.fitted_models:
            prediction = self.predict_fold(model, x_test)
            prediction = prediction.reshape(1, -1) if prediction.ndim == 1 else prediction[None, ...]
            predictions.append(prediction)
        result = np.concatenate(predictions, axis=0)[self.fold_assignments]
        return cast("np.ndarray[Any, np.dtype[np.floating[Any]]]", result)


class JackknifePlusRegressor(JackknifeCVBase, ConformalRegressor):
    """Jackknife+ Regressor."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int | Any | None = None,  # noqa: ANN401
        random_state: int | None = None,
        score_func: ScoreFunc | None = None,
        interval_func: IntervalFunc | None = None,
    ) -> None:
        """Initialize the JackknifePlusRegressor."""
        super().__init__(model_factory, cv, random_state)
        self.score_func = score_func
        self.interval_func = interval_func

    def predict_fold(self, model: Predictor, x: npt.NDArray) -> npt.NDArray[np.floating]:
        """Predict using the given model on the provided data."""
        prediction = model(x.tolist())
        return cast("npt.NDArray[np.floating]", self.to_numpy(prediction))

    def compute_scores(self, y_true: npt.NDArray, y_prediction: npt.NDArray) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores based on true and predicted values."""
        if self.score_func is not None:
            return cast(
                "npt.NDArray[np.floating]",
                np.asarray(self.score_func(y_true, y_prediction), dtype=float),
            )

        if y_prediction.ndim > 1 and y_prediction.shape[1] > 1:
            msg = "Residuals require scalar predictions. Provide a custom 'score_func' for other cases."
            raise ValueError(msg)

        y_true = y_true.astype(float).flatten()
        y_prediction = y_prediction.flatten()
        return cast("npt.NDArray[np.floating]", np.abs(y_true - y_prediction))

    def predict(self, x_test: Sequence[Any], alpha: float) -> npt.NDArray[np.floating]:
        """Predict prediction intervals for test data."""
        if not self.is_calibrated or self.nonconformity_scores is None:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        x_test_np = np.asarray(x_test)
        predictions = self.get_aligned_predictions(x_test_np)
        residuals = self.to_numpy(self.nonconformity_scores).reshape(-1, 1)

        # construct intervals
        if self.interval_func is not None:
            # custom logic (e.g. CQR / asymmetric)
            lower_bounds, upper_bounds = self.interval_func(predictions, residuals)
        else:
            # default logic (symmetric)
            if predictions.ndim > 2:
                msg = (
                    "Interval construction supports at most 2D predictions. "
                    "Provide a custom 'interval_func' for other cases."
                )
                raise ValueError(msg)
            lower_bounds = predictions - residuals
            upper_bounds = predictions + residuals
        lower = np.quantile(lower_bounds, alpha, axis=0, method="inverted_cdf")
        upper = np.quantile(upper_bounds, 1.0 - alpha, axis=0, method="inverted_cdf")
        return np.column_stack([lower, upper])


class JackknifePlusClassifier(JackknifeCVBase, ConformalClassifier):
    """Jackknife+ Classifier."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int | Any | None = None,  # noqa: ANN401
        random_state: int | None = None,
        use_accretive: bool = False,
        score_func: ScoreFunc | None = None,
    ) -> None:
        """Initialize the JackknifePlusClassifier."""
        super().__init__(model_factory, cv, random_state)
        self.use_accretive = use_accretive
        self.score_func = score_func
        self.classes: npt.NDArray | None = None

    def predict_fold(self, model: Predictor, x: npt.NDArray) -> npt.NDArray[np.floating]:
        """Predict using the given model on the provided data."""
        return cast("npt.NDArray[np.floating]", self.to_numpy(predict_probs(model, x)))

    def compute_scores(self, y_true: npt.NDArray, y_prediction: npt.NDArray) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores based on true and predicted values."""
        if self.classes is None:
            self.classes = np.unique(y_true)

        if self.score_func is not None:
            return cast(
                "npt.NDArray[np.floating]",
                np.asarray(self.score_func(y_true, y_prediction), dtype=float),
            )

        label_to_indices = {label: i for i, label in enumerate(self.classes)}
        y_indices = np.array([label_to_indices.get(y, -1) for y in y_true], dtype=int)
        if np.any(y_indices == -1):
            unknown_labels = {y for y, idx in zip(y_true, y_indices, strict=False) if idx == -1}
            msg = f"Unknown labels in y_true during calibration: {unknown_labels}."
            raise ValueError(msg)
        sample_indices = np.arange(len(y_true))
        return np.asarray(1.0 - y_prediction[sample_indices, y_indices], dtype=float)

    def predict(self, x_test: Sequence[Any], alpha: float, _probs: Any = None) -> npt.NDArray[np.bool_]:  # noqa: ANN401
        """Predict prediction sets for test data."""
        if not self.is_calibrated or self.nonconformity_scores is None:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        x_test_np = np.asarray(x_test)
        n_test = len(x_test_np)
        if self.classes is None:
            msg = "Classes are not defined. Ensure calibration has been performed."
            raise RuntimeError(msg)
        n_classes = len(self.classes)
        n_cal = len(self.nonconformity_scores)

        probs_aligned = self.get_aligned_predictions(x_test_np)
        nonconformity_scores_np = self.to_numpy(self.nonconformity_scores)
        calibration_scores_broadcasted = nonconformity_scores_np[:, None]
        prediction_sets = np.zeros((n_test, n_classes), dtype=bool)
        required_count = (1.0 - alpha) * (n_cal + 1)
        for calibration_index in range(n_classes):
            calibration_label = self.classes[calibration_index]
            if self.score_func is not None:
                flat_probs = probs_aligned.reshape(-1, n_classes)
                flat_labels = np.full(flat_probs.shape[0], calibration_label)
                flat_scores = self.score_func(flat_labels, flat_probs)
                score_test_calibration = flat_scores.reshape(n_cal, n_test)
            else:
                score_test_calibration = 1.0 - probs_aligned[:, :, calibration_index]
            conformity_mask = calibration_scores_broadcasted >= score_test_calibration
            count_conform = np.sum(conformity_mask, axis=0)
            prediction_sets[:, calibration_index] = count_conform >= required_count
        if self.use_accretive:
            mean_probs = np.mean(probs_aligned, axis=0)
            prediction_sets = accretive_completion(prediction_sets, mean_probs)
        return prediction_sets
