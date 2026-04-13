"""Conformalized prediction with credal sets as output."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from probly.method.method import predictor_transformation
from probly.predictor import Predictor, ProbabilisticClassifier, predict

if TYPE_CHECKING:
    from collections.abc import Iterable


@runtime_checkable
class NonconformityMeasure(Protocol):
    """Protocol for nonconformity measures."""

    def score(self, pred: np.ndarray, actual: np.ndarray) -> float:
        """Compute the nonconformity score between two distributions.

        Args:
            pred: Predicted distribution.
            actual: Actual distribution.

        Returns:
            The nonconformity score.
        """
        ...

    def score_vectorized(self, pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Compute the nonconformity scores between batches of distributions.

        Args:
            pred: Predicted distributions.
            actual: Actual distributions.

        Returns:
            The nonconformity scores.
        """
        ...


class TotalVariationMeasure:
    """Total Variation distance measure as nonconformity function."""

    def score(self, pred: np.ndarray, actual: np.ndarray) -> float:
        """Compute the total variation distance.

        Args:
            pred: Predicted distribution.
            actual: Actual distribution.

        Returns:
            The total variation distance.
        """
        return 0.5 * float(np.sum(np.abs(pred - actual)))

    def score_vectorized(self, pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Compute the total variation distance between batches of distributions.

        Args:
            pred: Predicted distributions.
            actual: Actual distributions.

        Returns:
            An array of total variation distances.
        """
        pred = np.atleast_2d(pred)
        actual = np.atleast_2d(actual)
        return 0.5 * np.sum(np.abs(pred - actual), axis=1)


class SimplexGenerator:
    """Generator for a simplex grid with the probability distributions."""

    def __init__(self, k_classes: int, resolution: float = 0.05) -> None:
        """Initialize the simplex generator.

        Args:
            k_classes: Number of classes.
            resolution: Steps for probabilities.
        """
        self.k = k_classes
        self.resolution = resolution
        self.grid = self._generate_grid()

    def _generate_grid(self) -> np.ndarray:
        """Generates the simplex grid.

        Returns:
            The generated simplex grid.

        Raises:
            ValueError: If k_classes > 4.
        """
        if self.k > 4:
            msg = f"Too many classes for grid generation. Max is 4, got {self.k}"
            raise ValueError(msg)

        bins = int(1.0 / self.resolution)
        ranges = [np.arange(0, bins + 1) for _ in range(self.k)]
        combinations = np.array(list(product(*ranges)))
        valid_mask = np.sum(combinations, axis=1) == bins
        valid_combinations = combinations[valid_mask]

        simplex_grid = valid_combinations * self.resolution
        simplex_grid = simplex_grid / np.sum(simplex_grid, axis=1, keepdims=True)
        return simplex_grid

    def get_grid(self) -> np.ndarray:
        """Returns the generated grid.

        Returns:
            The simplex grid.
        """
        return self.grid


class ConformalCredalSet:
    """A set of distributions defined by a distance threshold.

    Attributes:
        prediction: The nominal probability distribution.
        threshold: The distance threshold.
        measure: The nonconformity measure used.
        valid_grid: Optional grid of points within the threshold on the simplex.
    """

    def __init__(
        self,
        prediction: np.ndarray,
        threshold: float,
        measure: NonconformityMeasure,
        valid_grid: np.ndarray | None = None,
    ) -> None:
        """Initialize the credal set.

        Args:
            prediction: Predicted distribution.
            threshold: Distance threshold.
            measure: Distance measure.
            valid_grid: Optional grid of valid points.
        """
        self.prediction = prediction
        self.threshold = threshold
        self.measure = measure
        self.valid_grid = valid_grid


@runtime_checkable
class ConformalCredalPredictor[**In, Out](Predictor[In, Out], Protocol):
    """Protocol for conformalized credal predictors."""

    def calibrate(self, x_calib: Iterable[Any], y_calib: Iterable[Any], alpha: float = 0.1) -> float:
        """Calibrate the predictor.

        Args:
            x_calib: Calibration inputs.
            y_calib: Calibration targets.
            alpha: Significance level.

        Returns:
            The calibrated threshold.
        """
        ...

    @property
    def is_calibrated(self) -> bool:
        """Check if the predictor is calibrated.

        Returns:
            True if calibrated, False otherwise.
        """
        ...


class ConformalCredalSetPredictor[**In]:
    """Implementation of a conformal credal predictor."""

    def __init__(
        self,
        model: Predictor[In, Any],
        k_classes: int,
        measure: NonconformityMeasure | None = None,
        resolution: float = 0.05,
    ) -> None:
        """Initialize the conformal credal predictor.

        Args:
            model: The base model to be used.
            k_classes: Number of classes.
            measure: The distance measure to use.
            resolution: Resolution for the simplex grid.
        """
        self.model = model
        self.measure = measure or TotalVariationMeasure()
        self.k = k_classes
        self.calibrated_threshold: float | None = None

        self.generator: SimplexGenerator | None = None
        if self.k <= 4:
            self.generator = SimplexGenerator(k_classes=self.k, resolution=resolution)
            self.simplex_grid = self.generator.get_grid()

    def calibrate(self, x_calib: Iterable[Any], y_calib: Iterable[Any], alpha: float = 0.1) -> float:
        """Calibrate the threshold on a calibration set.

        Args:
            x_calib: Calibration input data.
            y_calib: Calibration labels.
            alpha: Significance level.

        Returns:
            The calibrated threshold.
        """
        probs = np.asarray(predict(self.model, x_calib))
        y_calib_np = np.asarray(y_calib)

        if y_calib_np.ndim == 1:
            n_samples = len(y_calib_np)
            y_one_hot = np.zeros((n_samples, self.k))
            y_one_hot[np.arange(n_samples), y_calib_np.astype(int)] = 1.0
            y_calib_np = y_one_hot

        scores = self.measure.score_vectorized(probs, y_calib_np)

        n = len(scores)
        alpha_prime = np.ceil((n + 1) * (1 - alpha)) / n
        alpha_prime = min(alpha_prime, 1.0)

        self.calibrated_threshold = float(np.quantile(scores, alpha_prime, method="higher"))
        return self.calibrated_threshold

    def conformal_predict(self, *args: In.args, **kwargs: In.kwargs) -> list[ConformalCredalSet]:
        """Predict credal sets for new inputs.

        Args:
            *args: Positional arguments for the base predictor.
            **kwargs: Keyword arguments for the base predictor.

        Returns:
            The predicted credal sets.

        Raises:
            RuntimeError: If the predictor is not calibrated.
        """
        if self.calibrated_threshold is None:
            msg = "Predictor must be calibrated before calling predict()."
            raise RuntimeError(msg)

        preds = predict(self.model, *args, **kwargs)
        probs = np.asarray(preds)

        credal_sets: list[ConformalCredalSet] = []
        for p in probs:
            valid_grid = None

            if self.generator is not None:
                grid_scores = self.measure.score_vectorized(p, self.simplex_grid)
                valid_mask = grid_scores <= self.calibrated_threshold
                valid_grid = self.simplex_grid[valid_mask]

            c_set = ConformalCredalSet(
                prediction=p,
                threshold=self.calibrated_threshold,
                measure=self.measure,
                valid_grid=valid_grid,
            )

            credal_sets.append(c_set)
        return credal_sets

    @property
    def is_calibrated(self) -> bool:
        """Check if the predictor is calibrated.

        Returns:
            True if calibrated, False otherwise.
        """
        return self.calibrated_threshold is not None


@predictor_transformation(permitted_predictor_types=(ProbabilisticClassifier,))
@ConformalCredalPredictor.register_factory
def conformal_credal_prediction[T: Predictor](
    base: T,
    k_classes: int,
    measure: NonconformityMeasure | None = None,
    resolution: float = 0.05,
) -> ConformalCredalPredictor:
    """Create a conformal credal predictor from a base predictor.

    Args:
        base: The base model to be used.
        k_classes: Number of classes.
        measure: The distance measure to use.
        resolution: Resolution for the simplex grid.

    Returns:
        The conformal credal predictor.
    """
    return ConformalCredalSetPredictor(model=base, k_classes=k_classes, measure=measure, resolution=resolution)  # type: ignore[return-value]
