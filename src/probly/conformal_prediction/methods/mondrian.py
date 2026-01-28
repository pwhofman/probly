"""Mondrian and Class-Conditional Conformal Prediction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
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

# Group functions
RegionFunc = Callable[[Sequence[Any]], npt.NDArray[np.int_] | Sequence[int]]
ClassFunc = Callable[[Sequence[Any], Sequence[Any] | None], npt.NDArray[np.int_] | Sequence[int]]


class GroupedConformalBase(ConformalPredictor):
    """Base class for group based conformal prediction (Mondrian & Class-Conditional).

    Groups can be regions (Mondrian) or classes (ClassConditional).
    Shared calibrate() logic; subclasses implement predict().
    """

    score: Score
    group_func: ClassFunc

    # thresholds per group
    group_thresholds: dict[int, float | np.floating]
    group_thresholds_lower: dict[int, float | np.floating]
    group_thresholds_upper: dict[int, float | np.floating]
    is_asymmetric: bool

    def __init__(self, model: Predictor, group_func: ClassFunc) -> None:
        """Initialize base conformal predictor."""
        super().__init__(model=model)
        self.group_func = group_func
        self.group_thresholds = {}
        self.group_thresholds_lower = {}
        self.group_thresholds_upper = {}
        self.is_asymmetric = False

    @staticmethod
    def to_numpy(data: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Convert tensor or array-like to numpy array."""
        if torch is not None and isinstance(data, Tensor):
            return cast("npt.NDArray[np.floating]", data.detach().cpu().numpy())
        return np.asarray(data, dtype=float)

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate group-wise thresholds on a calibration dataset."""
        # compute nonconformity scores
        nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)
        scores_np = self.to_numpy(nonconformity_scores)

        # get group ids for each sample
        group_ids = self.group_func(x_cal, y_cal)
        group_ids_np = np.asarray(group_ids, dtype=int)

        if group_ids_np.shape[0] != scores_np.shape[0]:
            msg = f"Group ids and scores must have same length, got {group_ids_np.shape[0]} vs {scores_np.shape[0]}"
            raise ValueError(msg)

        # compute threshold per group
        unique_groups = np.unique(group_ids_np)

        if scores_np.ndim == 1 or (scores_np.ndim == 2 and scores_np.shape[1] == 1):
            # symmetric
            self.is_asymmetric = False
            self.group_thresholds = {}
            scores_flat = scores_np.flatten()

            for group_id in unique_groups:
                mask = group_ids_np == group_id
                scores_in_group = scores_flat[mask]

                if scores_in_group.size > 0:
                    self.group_thresholds[int(group_id)] = calculate_quantile(scores_in_group, alpha)

        elif scores_np.ndim == 2 and scores_np.shape[1] == 2:
            # asymmetric (for CQR)
            self.is_asymmetric = True
            self.group_thresholds_lower = {}
            self.group_thresholds_upper = {}

            alpha_lower = alpha / 2
            alpha_upper = 1 - alpha / 2

            for group_id in unique_groups:
                mask = group_ids_np == group_id
                scores_lower = scores_np[mask, 0]
                scores_upper = scores_np[mask, 1]

                if scores_lower.size > 0:
                    self.group_thresholds_lower[int(group_id)] = calculate_quantile(scores_lower, alpha_lower)
                    self.group_thresholds_upper[int(group_id)] = calculate_quantile(scores_upper, alpha_upper)
        else:
            msg = f"Score shape {scores_np.shape} not supported. Expected 1D or 2D."
            raise ValueError(msg)

        self.is_calibrated = True
        self.threshold = None
        return alpha

    def _get_thresholds_per_sample(
        self,
        group_ids_np: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.floating]:
        """Assign threshold to each sample based on group membership."""
        n_samples = group_ids_np.shape[0]
        thresholds = np.empty(n_samples, dtype=float)

        # use max threshold as fallback for not calibrated groups
        if not self.group_thresholds:
            msg = "No groups calibrated. Call calibrate() before prediction."
            raise RuntimeError(msg)

        # fallback: use max threshold among calibrated groups
        max_threshold = max(self.group_thresholds.values())  # type: ignore[type-var]

        for i, group_id in enumerate(group_ids_np):
            thresholds[i] = self.group_thresholds.get(int(group_id), max_threshold)

        return thresholds

    def _get_thresholds_per_sample_asym(
        self,
        group_ids_np: npt.NDArray[np.int_],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Assign asymmetric thresholds (lower/upper) per sample."""
        n_samples = group_ids_np.shape[0]
        threshold_lower = np.empty(n_samples, dtype=float)
        threshold_upper = np.empty(n_samples, dtype=float)

        max_lower = max(self.group_thresholds_lower.values()) if self.group_thresholds_lower else np.inf  # type: ignore[type-var]
        max_upper = max(self.group_thresholds_upper.values()) if self.group_thresholds_upper else np.inf  # type: ignore[type-var]

        for i, group_id in enumerate(group_ids_np):
            gid = int(group_id)
            threshold_lower[i] = self.group_thresholds_lower.get(gid, max_lower)
            threshold_upper[i] = self.group_thresholds_upper.get(gid, max_upper)

        return threshold_lower, threshold_upper


class MondrianConformalClassifier(GroupedConformalBase, ConformalClassifier):
    """Mondrian (region-wise) conformal predictor for classification."""

    score: ClassificationScore

    def __init__(
        self,
        model: Predictor,
        score: ClassificationScore,
        region_func: RegionFunc,
        use_accretive: bool = False,
    ) -> None:
        """Create Mondrian conformal predictor for classification."""
        # wrap region_func to match ClassFunc signature
        super().__init__(model=model, group_func=lambda x, _y=None: region_func(x))
        self.score = score
        self.use_accretive = use_accretive

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return Mondrian prediction sets."""
        if not self.is_calibrated or not self.group_thresholds:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        scores = self.score.predict_nonconformity(x_test)
        scores_np = self.to_numpy(scores)
        n_test, _ = scores_np.shape

        # get region ids for test samples
        region_ids = self.group_func(x_test, None)
        region_ids_np = np.asarray(region_ids, dtype=int)

        if region_ids_np.shape[0] != n_test:
            msg = "Region ids must match test size."
            raise ValueError(msg)

        # assign threshold per sample
        thresholds_per_sample = self._get_thresholds_per_sample(region_ids_np)
        thresholds_per_sample = thresholds_per_sample.reshape(n_test, 1)

        prediction_sets = scores_np <= thresholds_per_sample

        if self.use_accretive:
            if probs is None:
                probs_raw: npt.NDArray[np.floating] = predict_probs(self.model, x_test)
                probs_np = self.to_numpy(probs_raw)
            else:
                probs_np = probs
            prediction_sets = accretive_completion(prediction_sets, probs_np)

        return prediction_sets.astype(bool)


class MondrianConformalRegressor(GroupedConformalBase, ConformalRegressor):
    """Mondrian (region-wise) conformal predictor for regression."""

    score: RegressionScore

    def __init__(
        self,
        model: Predictor,
        score: RegressionScore,
        region_func: RegionFunc,
    ) -> None:
        """Create Mondrian conformal predictor for regression."""
        super().__init__(model=model, group_func=lambda x, _y=None: region_func(x))
        self.score = score

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
    ) -> npt.NDArray[np.floating]:
        """Return Mondrian prediction intervals."""
        if not self.is_calibrated:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        y_hat = self.model(x_test)
        y_hat_np = self.to_numpy(y_hat)

        region_ids = self.group_func(x_test, None)
        region_ids_np = np.asarray(region_ids, dtype=int)

        if self.is_asymmetric:
            if y_hat_np.ndim != 2 or y_hat_np.shape[1] != 2:
                msg = "Asymmetric intervals expect model output shape (N, 2)."
                raise ValueError(msg)

            if not self.group_thresholds_lower or not self.group_thresholds_upper:
                msg = "Asymmetric thresholds not calibrated."
                raise RuntimeError(msg)

            threshold_lower, threshold_upper = self._get_thresholds_per_sample_asym(region_ids_np)

            lower = y_hat_np[:, 0] - threshold_lower
            upper = y_hat_np[:, 1] + threshold_upper
        else:
            y_hat_flat = y_hat_np.flatten()

            if not self.group_thresholds:
                msg = "Standard threshold is not calibrated."
                raise RuntimeError(msg)

            thresholds = self._get_thresholds_per_sample(region_ids_np)
            lower = y_hat_flat - thresholds
            upper = y_hat_flat + thresholds

        return cast("npt.NDArray[np.floating]", np.stack([lower, upper], axis=1))
