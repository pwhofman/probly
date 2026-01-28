# src/probly/conformal_prediction/scores/common.py
"""Common structures for conformal prediction scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.methods.common import predict_probs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import jax
    import jax.numpy as jnp
    import torch

    from probly.conformal_prediction.methods.common import Predictor

# PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# JAX / Flax
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class Score(Protocol):
    """Interface for nonconformity scores used in split conformal prediction.

    Each score (APS, LAC, RAPS, ...) must implement:
    - calibration_nonconformity: used on calibration data.
    - predict_nonconformity: used on test data, must return a score matrix
      of shape according to the specific score type (classification or regression).
    """

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return 1D array of nonconformity scores for calibration instances."""
        ...


class ClassificationScoreProtocol(Score, Protocol):
    """Nonconformity scores for classification tasks.

    calibration_nonconformity: 1D scores from Score.
    predict_nonconformity: 2D scores (n_instances, n_labels).
    """

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: Any | None = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Return 2D score matrix of shape (n_instances, n_labels)."""
        ...


class RegressionScoreProtocol(Score, Protocol):
    """Nonconformity scores for regression (e.g.. Residuals).

    calibration_nonconformity: 1D scores (|y - y_hat|, standardized Residuals, ...).
    predict_nonconformity: 1D scores or local scales (n_instances,).
    """

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> Any:  # noqa: ANN401
        """Return 1D scores or local scales of shape (n_instances,)."""
        ...

    def construct_intervals(
        self,
        y_hat: Any,  # noqa: ANN401
        threshold: float,
    ) -> Any:  # noqa: ANN401
        """Construct prediction intervals based on model output and threshold.

        Args:
            y_hat: Model output (n_samples, ) or (n_samples, 2) etc.
            threshold: Calibrated q-hat.

        Returns:
            Intervals as (n_samples, 2) matrix [lower, upper].
        """
        ...


class ClassificationScore(ClassificationScoreProtocol):
    """Generic implementation for classification scores.

    Handles APS, LAC, RAPS, SAPS by delegating to a score_func.
    Randomization must be built into the score_func if needed.
    """

    def __init__(
        self,
        model: Predictor,
        score_func: Callable[[Any], Any],
    ) -> None:
        """Initialize classification score.

        Args:
        model: The prediction model.
        score_func: Function that takes probabilities and returns score matrix.
                    Randomization (for APS/SAPS) MUST be built into this function.
        """
        self.model = model
        self.score_func = score_func

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        probs: Any | None = None,  # noqa: ANN401
    ) -> npt.NDArray[np.floating]:
        """Compute calibration scores (vectorized, backend-agnostic)."""
        # get probabilities (stays on GPU if using PyTorch)
        if probs is None:
            probs = predict_probs(self.model, x_cal)

        # compute scores using the score function
        all_scores = self.score_func(probs)

        # extract scores for true labels (efficient, backend-aware)

        # PyTorch
        if TORCH_AVAILABLE and isinstance(all_scores, torch.Tensor):
            return self._extract_torch_scores(all_scores, y_cal)

        # JAX / Flax
        if JAX_AVAILABLE and isinstance(all_scores, jax.Array):
            return self._extract_jax_scores(all_scores, y_cal)

        # NumPy fallback
        return self._extract_numpy_scores(all_scores, y_cal)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: Any | None = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Compute scores for all labels (stays on original device)."""
        if probs is None:
            probs = predict_probs(self.model, x_test)

        # return the matrix of scores
        return self.score_func(probs)

    def _extract_numpy_scores(self, scores: Any, y: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Extract true label scores using NumPy (vectorized)."""
        scores_np = np.asarray(scores, dtype=float)
        labels_np = np.asarray(y, dtype=int)

        result = scores_np[np.arange(len(labels_np)), labels_np]
        return cast("npt.NDArray[np.floating]", result.astype(float))

    def _extract_torch_scores(self, scores: torch.Tensor, y: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Extract true label scores using PyTorch (GPU-friendly)."""
        device = scores.device

        # convert labels to tensor if needed
        if isinstance(y, torch.Tensor):
            labels = y.to(device).long()
        else:
            labels = torch.as_tensor(y, dtype=torch.long, device=device)

        # gather scores efficiently on GPU
        true_scores = scores.gather(1, labels.view(-1, 1)).squeeze(1)

        # convert to numpy at the end
        return true_scores.cpu().numpy()

    def _extract_jax_scores(self, scores: jax.Array, y: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401
        """Extract true label scores using JAX (TPU/GPU-friendly)."""
        # ensure labels are JAX array
        labels = jnp.asarray(y, dtype=int)

        # JAX Advanced Indexing works similarly to NumPy
        n = scores.shape[0]
        true_scores = scores[jnp.arange(n), labels]

        # convert to numpy for quantile calculation
        result = np.asarray(true_scores, dtype=float)
        return result


class RegressionScore(RegressionScoreProtocol):
    """Generic implementation for regression scores.

    Handles AbsoluteError, CQR, etc.
    """

    def __init__(
        self,
        model: Predictor,
        score_func: Callable[[Any, Any], Any],
        interval_func: Callable[[Any, float], Any] | None = None,
    ) -> None:
        """Initialize regression score.

        Args:
        model: The prediction model.
        score_func: Function that computes scores from (predictions, true_values).
        interval_func: Optional function to construct intervals.
        """
        self.model = model
        self.score_func = score_func
        self.interval_func = interval_func

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Compute calibration scores."""
        # get predictions (stays on device)
        predictions = self.model(x_cal)

        # PyTorch
        if TORCH_AVAILABLE and isinstance(predictions, torch.Tensor):
            y_cal_tensor = torch.as_tensor(y_cal, device=predictions.device)
            scores = self.score_func(y_cal_tensor, predictions)
            return scores.cpu().numpy()

        # JAX / Flax
        if JAX_AVAILABLE and isinstance(predictions, jax.Array):
            y_cal_jax = jnp.asarray(y_cal)
            scores = self.score_func(y_cal_jax, predictions)
            return np.asarray(scores, dtype=float)

        # NumPy fallback
        predictions_np = np.asarray(predictions, dtype=float)
        y_cal_np = np.asarray(y_cal, dtype=float)
        scores = self.score_func(y_cal_np, predictions_np)
        return np.asarray(scores, dtype=float)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> Any:  # noqa: ANN401
        """For regression, return predictions for interval construction."""
        return self.model(x_test)

    def construct_intervals(
        self,
        y_hat: Any,  # noqa: ANN401
        threshold: float,
    ) -> Any:  # noqa: ANN401
        """Construct prediction intervals (Preserves backend)."""
        if self.interval_func is not None:
            return self.interval_func(y_hat, threshold)

        # PyTorch
        if TORCH_AVAILABLE and isinstance(y_hat, torch.Tensor):
            if y_hat.ndim == 2 and y_hat.shape[1] == 2:
                # handle 2D case for intervals (asymmetric intervals, (N,2))
                lower_torch = y_hat[:, 0] - threshold
                upper_torch = y_hat[:, 1] + threshold
            else:
                # handle 1D case (symmetric intervals, (N,))
                lower_torch = y_hat - threshold
                upper_torch = y_hat + threshold

            return torch.stack([lower_torch, upper_torch], dim=1)

        # JAX / Flax
        if JAX_AVAILABLE and isinstance(y_hat, jax.Array):
            if y_hat.ndim == 2 and y_hat.shape[1] == 2:
                # handle 2D case for intervals (asymmetric intervals, (N,2))
                lower_jax = y_hat[:, 0] - threshold
                upper_jax = y_hat[:, 1] + threshold
            else:
                # handle 1D case (symmetric intervals, (N,))
                lower_jax = y_hat - threshold
                upper_jax = y_hat + threshold

            return jnp.stack([lower_jax, upper_jax], axis=1)

        # NumPy fallback
        y_hat_np = np.asarray(y_hat, dtype=float)

        if y_hat_np.ndim == 2 and y_hat_np.shape[1] == 2:
            # handle 2D case for intervals (asymmetric intervals, (N,2))
            lower_np = y_hat_np[:, 0] - threshold
            upper_np = y_hat_np[:, 1] + threshold
        else:
            # handle 1D case (symmetric intervals, (N,))
            if y_hat_np.ndim > 1:
                y_hat_np = y_hat_np.flatten()
            lower_np = y_hat_np - threshold
            upper_np = y_hat_np + threshold

        return np.stack([lower_np, upper_np], axis=1)
