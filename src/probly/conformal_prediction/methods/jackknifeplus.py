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

# Type Definitions
ScoreFunc = Callable[[npt.NDArray, npt.NDArray], npt.NDArray[np.floating]]
IntervalFunc = Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]


class JackknifeCVBase(ConformalPredictor):
    """Base class for resampling-based conformal prediction (Jackknife+ / CV+)."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int | None = None,
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

        for idx, prediction in oof_predictions_list:
            oof_predictions[idx] = prediction

        self.nonconformity_scores = self.compute_scores(y_np, oof_predictions)
        scores_np = self.to_numpy(self.nonconformity_scores)

        self.threshold = calculate_quantile(scores_np, alpha)
        self.is_calibrated = True
        return self.threshold

    def get_aligned_predictions(self, x_test: npt.NDArray) -> npt.NDArray[np.floating]:
        """Get predictions from each fold model aligned to original data order."""
        if self.fold_assignments is None:
            msg = "Fold assignments are not defined. Ensure calibration has been performed."
            raise RuntimeError(msg)

        n_cal = len(self.fold_assignments)
        n_test = len(x_test)

        # get predictions from each model
        predictions = []
        for model in self.fitted_models:
            pred = self.predict_fold(model, x_test)
            predictions.append(pred)  # each: (n_test,) oder (n_test, n_classes)

        # determine shape
        first_pred = predictions[0]
        if first_pred.ndim == 1:
            aligned = np.zeros((n_cal, n_test), dtype=float)
        else:
            n_classes = first_pred.shape[1]
            aligned = np.zeros((n_cal, n_test, n_classes), dtype=float)

        # for each calibration sample: align predictions according to fold assignments
        for cal_idx, fold_idx in enumerate(self.fold_assignments):
            if fold_idx < len(predictions):
                aligned[cal_idx] = predictions[fold_idx]

        return aligned


class JackknifePlusRegressor(JackknifeCVBase, ConformalRegressor):
    """Jackknife+ Regressor."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int | None = None,
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

    def compute_scores(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores based on true and predicted values."""
        if self.score_func is not None:
            return cast(
                "npt.NDArray[np.floating]",
                np.asarray(self.score_func(y_true, y_pred), dtype=float),
            )

        y_true = y_true.astype(float).flatten()

        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            lower_q = y_pred[:, 0]
            upper_q = y_pred[:, 1]
            diff_lower = lower_q - y_true
            diff_upper = y_true - upper_q
            result = np.maximum(diff_lower, diff_upper)
            return cast("npt.NDArray[np.floating]", result)

        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            msg = "Residuals require scalar predictions. Provide a custom 'score_func' for other cases."
            raise ValueError(msg)

        y_pred = y_pred.flatten()
        return cast("npt.NDArray[np.floating]", np.abs(y_true - y_pred))

    def predict(self, x_test: Sequence[Any], alpha: float) -> npt.NDArray[np.floating]:
        """Predict prediction intervals for test data."""
        if not self.is_calibrated or self.nonconformity_scores is None:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        x_test_np = np.asarray(x_test)

        aligned_predictions = self.get_aligned_predictions(x_test_np)  # (n_cal, n_test)

        scores = self.to_numpy(self.nonconformity_scores).reshape(-1, 1)

        # construct intervals
        if self.interval_func is not None:
            # custom logic (e.g. CQR / asymmetric)
            lower_bounds, upper_bounds = self.interval_func(aligned_predictions, scores)
        else:
            # default logic (symmetric)
            lower_bounds = aligned_predictions - scores
            upper_bounds = aligned_predictions + scores

        # compute 1 - alpha quantiles for all K folds
        lower = np.quantile(lower_bounds, alpha, axis=0, method="inverted_cdf")
        upper = np.quantile(upper_bounds, 1.0 - alpha, axis=0, method="inverted_cdf")
        return np.column_stack([lower, upper])


class JackknifePlusClassifier(JackknifeCVBase, ConformalClassifier):
    """Jackknife+ Classifier."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int | None = None,
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

    def compute_scores(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores based on true and predicted values."""
        # determine classes if not set
        if self.classes is None:
            self.classes = np.unique(y_true)

        if self.score_func is not None:
            return cast(
                "npt.NDArray[np.floating]",
                np.asarray(self.score_func(y_true, y_pred), dtype=float),
            )
        # map true labels to indices
        label_to_idx = {label: i for i, label in enumerate(self.classes)}
        y_idx = np.array([label_to_idx.get(y, -1) for y in y_true], dtype=int)

        # check for unknown labels
        if np.any(y_idx == -1):
            unknown_labels = {y for y, idx in zip(y_true, y_idx, strict=False) if idx == -1}
            msg = f"Unknown labels in y_true during calibration: {unknown_labels}."
            raise ValueError(msg)

        # nonconformity score: 1 - predicted probability of the true class
        sample_idx = np.arange(len(y_true))
        return np.asarray(1.0 - y_pred[sample_idx, y_idx], dtype=float)

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

        # LOO predictions aligned to folds (n_cal, n_test, n_classes)
        probs_aligned = self.get_aligned_predictions(x_test_np)

        scores_cal_np = self.to_numpy(self.nonconformity_scores)
        cal_scores = scores_cal_np[:, None]  # shape (n_cal, 1)

        prediction_sets = np.zeros((n_test, n_classes), dtype=bool)
        # compute the required count threshold
        required_count = (1.0 - alpha) * (n_cal + 1)
        required_count = max(0, required_count)

        for class_idx in range(n_classes):
            label = self.classes[class_idx]

            # compute test score thresholds for the current class
            if self.score_func is not None:
                # reshape to (n_cal * n_test, n_classes)
                flat_probs = probs_aligned.reshape(-1, n_classes)
                flat_labels = np.full(flat_probs.shape[0], label)
                flat_scores = self.score_func(flat_labels, flat_probs)
                # reshape back to (n_cal, n_test)
                test_score_threshold = flat_scores.reshape(n_cal, n_test)
            else:
                test_score_threshold = 1.0 - probs_aligned[:, :, class_idx]

            # compare calibration scores to test thresholds
            conformity_mask = cal_scores >= test_score_threshold
            count_conform = np.sum(conformity_mask, axis=0)
            # fill prediction sets
            prediction_sets[:, class_idx] = count_conform >= required_count

        if self.use_accretive:
            # average probabilities across folds for accretive completion
            mean_probs = np.mean(probs_aligned, axis=0)
            prediction_sets = accretive_completion(prediction_sets, mean_probs)

        return prediction_sets
