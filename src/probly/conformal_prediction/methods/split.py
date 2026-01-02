"""Split conformal prediction methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.conformal_prediction.scores.common import Score


import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from probly.conformal_prediction.methods.common import ConformalPredictor, Predictor, predict_probs
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
    """Generic split conformal predictor for classification."""

    def __init__(
        self,
        model: Predictor,
        score: Score,
        use_accretive: bool = False,
    ) -> None:
        """Create a split conformal predictor."""
        super().__init__(model=model)
        self.score = score
        self.use_accretive = use_accretive

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
        scores_for_quantile = self.nonconformity_scores
        if torch is not None and isinstance(scores_for_quantile, Tensor):
            scores_for_quantile = scores_for_quantile.detach().cpu().numpy()
        else:
            # covers numpy + jax
            scores_for_quantile = np.asarray(scores_for_quantile)

        # calculate quantile threshold
        self.threshold = calculate_quantile(scores_for_quantile, alpha)

        self.is_calibrated = True
        return self.threshold

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

        # compute scores (either using existing probs or predicting new ones inside score)
        scores = self.score.predict_nonconformity(x_test)  # shape: matrix (n_instances, n_labels)

        if scores.ndim != 2:
            msg = "predict_nonconformity must return 2D-Matrix (n_instances, n_labels)."
            raise ValueError(msg)

        # convert scores to Numpy for comparison with threshold
        if torch is not None and isinstance(scores, Tensor):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores)

        # sets defined: label included when score <= threshold
        prediction_sets = scores_np <= self.threshold  # bool-Array (n_instances, n_labels)

        # accretive completion for empty sets
        if self.use_accretive:
            probs = predict_probs(self.model, x_test)

            if torch is not None and isinstance(probs, Tensor):
                probs = probs.detach().cpu().numpy()

            probs_np = np.asarray(probs)
            prediction_sets = accretive_completion(prediction_sets, probs_np)

        return prediction_sets
