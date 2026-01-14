"""Split conformal prediction methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.conformal_prediction.scores.common import ClassificationScore, RegressionScore, Score


import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from probly.conformal_prediction.methods.common import (
    ConformalClassifier,
    ConformalPredictor,
    ConformalRegressor,
    Predictor,
    predict_probs,
)
from probly.conformal_prediction.scores.lac.common import accretive_completion
from probly.conformal_prediction.utils.quantile import calculate_quantile


class SplitConformal:
    """Utility to split data into training and calibration sets."""

    def __init__(
        self,
        calibration_ratio: float = 0.3,
        random_state: int | None = None,
    ) -> None:
        """Initialize the SplitConformal helper."""
        self.calibration_ratio = calibration_ratio
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.train_indices: npt.NDArray[np.int_] | None = None
        self.cal_indices: npt.NDArray[np.int_] | None = None

    def split(
        self,
        x: Sequence[Any],
        y: Sequence[Any],
        calibration_ratio: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and calibration sets."""
        ratio = calibration_ratio if calibration_ratio is not None else self.calibration_ratio

        if not 0 < ratio < 1:
            msg = f"calibration_ratio must be in (0, 1), got {ratio}"
            raise ValueError(msg)

        if len(x) < 2:
            msg = f"Need at least 2 samples, got {len(x)}"
            raise ValueError(msg)

        if len(x) != len(y):
            msg = f"x and y must have the same length. Got x: {len(x)}, y: {len(y)}"
            raise ValueError(msg)

        # convert to numpy arrays
        x_np = np.asarray(x)
        y_np = np.asarray(y)

        n_samples = len(x_np)
        indices = np.arange(n_samples)
        shuffled = self.rng.permutation(indices)

        split_idx = int(n_samples * (1.0 - ratio))
        self.train_indices = shuffled[:split_idx]
        self.cal_indices = shuffled[split_idx:]

        return (
            x_np[self.train_indices],
            y_np[self.train_indices],
            x_np[self.cal_indices],
            y_np[self.cal_indices],
        )

    def __str__(self) -> str:
        """String representation with basic split information."""
        if self.train_indices is None or self.cal_indices is None:
            return f"SplitConformal(ratio={self.calibration_ratio}, random_state={self.random_state})"

        n_train = len(self.train_indices)
        n_cal = len(self.cal_indices)
        ratio_actual = n_cal / (n_train + n_cal)
        return (
            f"SplitConformal: {n_train} train, {n_cal} calibration "
            f"(ratio={ratio_actual:.3f}, target={self.calibration_ratio})"
        )


class SplitConformalPredictor(ConformalPredictor):
    """Generic split conformal predictor base class."""

    score: Score

    def __init__(
        self,
        model: Predictor,
    ) -> None:
        """Create a split conformal predictor."""
        super().__init__(model=model)

    @staticmethod
    def to_numpy(x: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Convert tensor to NumPy on CPU (float dtype)."""
        if torch is not None and isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate the predictor on a calibration dataset."""
        # nonconformity score from object
        self.nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)

        # ensure scores are on CPU/Numpy for quantile calculation
        scores_np = self.to_numpy(self.nonconformity_scores)

        # calculate quantile threshold
        self.threshold = calculate_quantile(scores_np, alpha)

        self.is_calibrated = True
        return self.threshold


class SplitConformalClassifier(SplitConformalPredictor, ConformalClassifier):
    """Generic split conformal predictor for classification."""

    score: ClassificationScore

    def __init__(
        self,
        model: Predictor,
        score: ClassificationScore,
        use_accretive: bool = False,
    ) -> None:
        """Create a split conformal predictor for classification."""
        super().__init__(model=model)
        self.score = score
        self.use_accretive = use_accretive

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
        probs: Any = None,  # noqa: ANN401
    ) -> npt.NDArray[np.bool_]:
        """Return prediction sets as a (n_instances, n_labels) 0/1-matrix."""
        if not self.is_calibrated or self.threshold is None:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        # compute scores for test instances
        scores = self.score.predict_nonconformity(x_test)  # shape: matrix (n_instances, n_labels)

        if scores.ndim != 2:
            msg = "predict_nonconformity must return 2D-Matrix (n_instances, n_labels)."
            raise ValueError(msg)

        # convert scores to NumPy for comparison with threshold
        scores_np = self.to_numpy(scores)

        # sets defined: label included when score <= threshold
        prediction_sets = scores_np <= self.threshold  # bool-Array (n_instances, n_labels)

        # accretive completion for empty sets
        if self.use_accretive:
            probs = predict_probs(self.model, x_test)
            probs_np = self.to_numpy(probs)
            prediction_sets = accretive_completion(prediction_sets, probs_np)

        return prediction_sets


class SplitConformalRegressor(SplitConformalPredictor, ConformalRegressor):
    """Generic split conformal predictor for regression."""

    score: RegressionScore

    def __init__(
        self,
        model: Predictor,
        score: RegressionScore,
    ) -> None:
        """Create a split conformal predictor for regression."""
        super().__init__(model=model)
        self.score = score
        self.is_asymmetric: bool = False
        self.threshold_lower: float | None = None
        self.threshold_upper: float | None = None

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate thresholds for regression (supports symmetric and CQR)."""
        self.nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)
        # ensure numpy array
        scores_np = SplitConformalPredictor.to_numpy(self.nonconformity_scores)

        # determine if symmetric or asymmetric thresholds
        if scores_np.ndim == 1 or (scores_np.ndim == 2 and scores_np.shape[1] == 1):
            # standard symmetric residuals: single threshold
            self.is_asymmetric = False
            scores_flat = scores_np.flatten()
            self.threshold = calculate_quantile(scores_flat, alpha)
            self.threshold_lower = None
            self.threshold_upper = None
        elif scores_np.ndim == 2 and scores_np.shape[1] == 2:
            # asymmetric residuals: two thresholds
            self.is_asymmetric = True
            alpha_tail = alpha / 2  # Bonferroni split per side
            scores_lower = scores_np[:, 0]
            scores_upper = scores_np[:, 1]
            self.threshold_lower = calculate_quantile(scores_lower, alpha_tail)
            self.threshold_upper = calculate_quantile(scores_upper, alpha_tail)
            self.threshold = None
        else:
            msg = f"Score shape {scores_np.shape} not supported. Expected (n,), (n,1), or (n,2)."
            raise ValueError(msg)

        self.is_calibrated = True
        # return any float
        return self.threshold if self.threshold is not None else alpha

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
    ) -> npt.NDArray[np.floating]:
        """Return prediction intervals as a (n_instances, 2)-matrix [lower, upper]."""
        if not self.is_calibrated:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        # get model predictions
        y_hat = self.model(x_test)

        # convert predictions to NumPy for interval construction
        y_hat_np = SplitConformalPredictor.to_numpy(y_hat)

        if self.is_asymmetric:
            # expect model to return (N, 2): [lower_quantile, upper_quantile]
            if y_hat_np.ndim != 2 or y_hat_np.shape[1] != 2:
                msg = f"Asymmetric intervals expect model output shape (N, 2), got {y_hat_np.shape}"
                raise ValueError(msg)
            if self.threshold_lower is None or self.threshold_upper is None:
                msg = "Asymmetric thresholds not calibrated."
                raise RuntimeError(msg)
            lower = y_hat_np[:, 0] - self.threshold_lower
            upper = y_hat_np[:, 1] + self.threshold_upper
            return np.stack([lower, upper], axis=1)

        # symmetric intervals via score helper
        if self.threshold is None:
            msg = "Symmetric threshold not calibrated."
            raise RuntimeError(msg)
        return self.score.construct_intervals(y_hat_np, self.threshold)
