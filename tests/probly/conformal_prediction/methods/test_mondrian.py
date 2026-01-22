"""Tests for Mondrian Conformal Prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from src.probly.conformal_prediction.methods.mondrian import (
    GroupedConformalBase,
    MondrianConformalClassifier,
    MondrianConformalRegressor,
    RegionFunc,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, n_classes: int = 3) -> None:
        """Initialize MockModel."""
        self.n_classes = n_classes
        self._mode = "classification"

    def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        # helper to determine length
        batch_size = len(x) if hasattr(x, "__len__") else 1

        if self._mode == "classification":
            # return numpy array directly to match type hint
            return np.ones((batch_size, self.n_classes), dtype=float) * 0.5
        return np.ones((batch_size, 1), dtype=float) * 0.5

    def predict_probs(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        batch_size = len(x) if hasattr(x, "__len__") else 1
        return np.ones((batch_size, self.n_classes), dtype=float) * 0.5


class MockClassificationScore:
    """Mock classification score function."""

    def __init__(self, n_classes: int = 3) -> None:
        """Initialize MockClassificationScore."""
        self.n_classes = n_classes

    def calibration_nonconformity(self, x_cal: Sequence[Any], _y_cal: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Return random nonconformity scores for calibration."""
        n_samples = len(x_cal)
        rng = np.random.default_rng(42)
        return rng.random((n_samples, self.n_classes))

    def predict_nonconformity(self, x_test: Sequence[Any]) -> npt.NDArray[np.floating]:
        n_samples = len(x_test)
        rng = np.random.default_rng(42)
        return rng.random((n_samples, self.n_classes))


class MockRegressionScore:
    """Mock regression score function."""

    def __init__(self, is_cqr: bool = False) -> None:
        """Initialize MockRegressionScore."""
        self.is_cqr = is_cqr

    def calibration_nonconformity(self, x_cal: Sequence[Any], _y_cal: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Return random nonconformity scores for regression calibration."""
        n_samples = len(x_cal)
        rng = np.random.default_rng(42)
        if self.is_cqr:
            return rng.random((n_samples, 2))
        return rng.random((n_samples, 1))

    # add predict_nonconformity for completeness as base class uses it sometimes
    def predict_nonconformity(self, x_test: Sequence[Any]) -> npt.NDArray[np.floating]:
        return self.calibration_nonconformity(x_test, [])


@pytest.fixture
def mock_classification_model() -> MockModel:
    return MockModel(n_classes=3)


@pytest.fixture
def mock_regression_model() -> MockModel:
    model = MockModel()
    model._mode = "regression"  # noqa: SLF001
    return model


@pytest.fixture
def mock_classification_score() -> MockClassificationScore:
    return MockClassificationScore(n_classes=3)


@pytest.fixture
def mock_regression_score() -> MockRegressionScore:
    return MockRegressionScore(is_cqr=False)


@pytest.fixture
def mock_cqr_score() -> MockRegressionScore:
    return MockRegressionScore(is_cqr=True)


@pytest.fixture
def simple_region_func() -> RegionFunc:
    def func(x: Sequence[Any], _y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
        regions = []
        for sample in x:
            val = (sample[0] if len(sample) > 0 else 0) if hasattr(sample, "__len__") else float(sample)
            if val < 0.3:
                regions.append(0)
            elif val < 0.6:
                regions.append(1)
            else:
                regions.append(2)
        return np.array(regions, dtype=int)

    return func


@pytest.fixture
def random_data() -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    rng = np.random.default_rng(42)
    x_cal = rng.random((100, 5))
    y_cal = rng.integers(0, 3, 100)
    x_test = rng.random((20, 5))
    return x_cal, y_cal, x_test


@pytest.fixture
def regression_data() -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    rng = np.random.default_rng(42)
    x_cal = rng.random((100, 5))
    y_cal = rng.random(100)
    x_test = rng.random((20, 5))
    return x_cal, y_cal, x_test


class TestGroupedConformalBase:
    def test_initialization(self, mock_classification_model: MockModel) -> None:
        base = GroupedConformalBase(
            model=mock_classification_model,
            group_func=lambda x, _y=None: np.zeros(len(x), dtype=int),
        )
        assert base.model == mock_classification_model
        assert not base.is_calibrated
        assert base.group_thresholds == {}

    def test_to_numpy_tensor(self) -> None:
        base = GroupedConformalBase(
            model=None,
            group_func=lambda x, _y=None: np.zeros(len(x), dtype=int),
        )
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = base.to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_numpy_array(self) -> None:
        base = GroupedConformalBase(
            model=None,
            group_func=lambda x, _y=None: np.zeros(len(x), dtype=int),
        )
        arr = np.array([1.0, 2.0, 3.0])
        result = base.to_numpy(arr)
        assert result is arr or np.array_equal(result, arr)


class TestMondrianConformalClassifier:
    def test_initialization(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
        simple_region_func: RegionFunc,
    ) -> None:
        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=simple_region_func,
            use_accretive=False,
        )

        assert classifier.model == mock_classification_model
        assert classifier.score == mock_classification_score
        assert not classifier.is_calibrated
        assert classifier.use_accretive is False

    def test_calibration(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
        simple_region_func: RegionFunc,
        random_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, _ = random_data

        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=simple_region_func,
        )

        alpha = 0.1
        result_alpha = classifier.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), alpha)

        assert classifier.is_calibrated
        assert result_alpha == alpha
        assert len(classifier.group_thresholds) > 0

        for region_id, threshold in classifier.group_thresholds.items():
            assert isinstance(region_id, (int, np.integer))
            assert isinstance(threshold, (float, np.floating))
            assert 0 <= threshold <= 1

    def test_calibration_validation(self) -> None:
        classifier = MondrianConformalClassifier(
            model=MockModel(),
            score=cast(Any, MockClassificationScore()),
            region_func=lambda x, _y=None: np.zeros(len(x), dtype=int),
        )

        x_cal = [[1, 2], [3, 4]]
        y_cal = [0]

        with pytest.raises(ValueError, match=".*"):
            classifier.calibrate(x_cal, y_cal, 0.1)

    def test_prediction_without_calibration(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
        simple_region_func: RegionFunc,
        random_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        _, _, x_test = random_data

        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=simple_region_func,
        )

        with pytest.raises(RuntimeError, match="must be calibrated"):
            classifier.predict(cast(Sequence[Any], x_test), 0.1)

    def test_prediction_shape_and_type(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
        simple_region_func: RegionFunc,
        random_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, x_test = random_data

        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=simple_region_func,
        )

        classifier.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)
        prediction_sets = classifier.predict(cast(Sequence[Any], x_test), 0.1)

        assert prediction_sets.shape[0] == len(x_test)
        assert prediction_sets.shape[1] == mock_classification_score.n_classes
        assert prediction_sets.dtype == np.bool_
        assert np.all(prediction_sets.sum(axis=1) >= 1)

    def test_prediction_with_accretive(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
        simple_region_func: RegionFunc,
        random_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, x_test = random_data

        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=simple_region_func,
            use_accretive=True,
        )

        classifier.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)

        prediction_sets = classifier.predict(cast(Sequence[Any], x_test), 0.1)
        assert prediction_sets.shape == (len(x_test), 3)

        rng = np.random.default_rng(42)
        probs = rng.random((len(x_test), 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        prediction_sets_with_probs = classifier.predict(cast(Sequence[Any], x_test), 0.1, probs=probs)
        assert prediction_sets_with_probs.shape == (len(x_test), 3)

    def test_empty_groups_calibration(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
    ) -> None:
        def region_func(x: Sequence[Any], _y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
            return np.zeros(len(x), dtype=int)

        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=region_func,
        )

        rng = np.random.default_rng(42)
        x_cal = rng.random((10, 5))
        y_cal = rng.integers(0, 3, 10)

        classifier.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)
        assert 0 in classifier.group_thresholds
        assert classifier.is_calibrated

    def test_edge_case_single_sample(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
        simple_region_func: RegionFunc,
    ) -> None:
        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=simple_region_func,
        )

        x_cal = [[0.1, 0.2]]
        y_cal = [0]

        classifier.calibrate(x_cal, y_cal, 0.1)

        x_test = [[0.15, 0.25]]
        prediction_sets = classifier.predict(x_test, 0.1)
        assert prediction_sets.shape == (1, 3)


class TestMondrianConformalRegressor:
    def test_initialization_symmetric(
        self,
        mock_regression_model: MockModel,
        mock_regression_score: MockRegressionScore,
        simple_region_func: RegionFunc,
    ) -> None:
        regressor = MondrianConformalRegressor(
            model=mock_regression_model,
            score=cast(Any, mock_regression_score),
            region_func=simple_region_func,
        )

        assert regressor.model == mock_regression_model
        assert regressor.score == mock_regression_score
        assert not regressor.is_calibrated
        assert not regressor.is_asymmetric

    def test_calibration_symmetric(
        self,
        mock_regression_model: MockModel,
        mock_regression_score: MockRegressionScore,
        simple_region_func: RegionFunc,
        regression_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, _ = regression_data

        regressor = MondrianConformalRegressor(
            model=mock_regression_model,
            score=cast(Any, mock_regression_score),
            region_func=simple_region_func,
        )

        alpha = 0.1
        result_alpha = regressor.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), alpha)

        assert regressor.is_calibrated
        assert result_alpha == alpha
        assert not regressor.is_asymmetric
        assert len(regressor.group_thresholds) > 0

    def test_calibration_asymmetric(
        self,
        mock_regression_model: MockModel,
        mock_cqr_score: MockRegressionScore,
        simple_region_func: RegionFunc,
        regression_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, _ = regression_data

        regressor = MondrianConformalRegressor(
            model=mock_regression_model,
            score=cast(Any, mock_cqr_score),
            region_func=simple_region_func,
        )

        alpha = 0.1
        result_alpha = regressor.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), alpha)

        assert regressor.is_calibrated
        assert result_alpha == alpha
        assert regressor.is_asymmetric
        assert len(regressor.group_thresholds_lower) > 0
        assert len(regressor.group_thresholds_upper) > 0

    def test_prediction_symmetric(
        self,
        mock_regression_model: MockModel,
        mock_regression_score: MockRegressionScore,
        simple_region_func: RegionFunc,
        regression_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, x_test = regression_data

        regressor = MondrianConformalRegressor(
            model=mock_regression_model,
            score=cast(Any, mock_regression_score),
            region_func=simple_region_func,
        )

        regressor.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)
        intervals = regressor.predict(cast(Sequence[Any], x_test), 0.1)

        assert intervals.shape[0] == len(x_test)
        assert intervals.shape[1] == 2
        assert np.all(intervals[:, 0] <= intervals[:, 1])
        assert np.issubdtype(intervals.dtype, np.floating)

    def test_prediction_asymmetric(
        self,
        mock_cqr_score: MockRegressionScore,
        simple_region_func: RegionFunc,
        regression_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, x_test = regression_data

        class MockCQRModel:
            def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
                batch_size = len(x)
                return np.ones((batch_size, 2)) * 0.5

        regressor = MondrianConformalRegressor(
            model=MockCQRModel(),
            score=cast(Any, mock_cqr_score),
            region_func=simple_region_func,
        )

        regressor.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)
        intervals = regressor.predict(cast(Sequence[Any], x_test), 0.1)

        assert intervals.shape[0] == len(x_test)
        assert intervals.shape[1] == 2
        assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_prediction_without_calibration(
        self,
        mock_regression_model: MockModel,
        mock_regression_score: MockRegressionScore,
        simple_region_func: RegionFunc,
        regression_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        _, _, x_test = regression_data

        regressor = MondrianConformalRegressor(
            model=mock_regression_model,
            score=cast(Any, mock_regression_score),
            region_func=simple_region_func,
        )

        with pytest.raises(RuntimeError, match="must be calibrated"):
            regressor.predict(cast(Sequence[Any], x_test), 0.1)

    def test_invalid_asymmetric_prediction(
        self,
        mock_regression_model: MockModel,
        mock_cqr_score: MockRegressionScore,
        simple_region_func: RegionFunc,
        regression_data: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]],
    ) -> None:
        x_cal, y_cal, x_test = regression_data

        regressor = MondrianConformalRegressor(
            model=mock_regression_model,  # returns (N, 1)
            score=cast(Any, mock_cqr_score),
            region_func=simple_region_func,
        )

        regressor.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)

        with pytest.raises(ValueError, match="Asymmetric intervals expect"):
            regressor.predict(cast(Sequence[Any], x_test), 0.1)

    def test_edge_case_empty_regions(
        self,
        mock_regression_model: MockModel,
        mock_regression_score: MockRegressionScore,
    ) -> None:
        def region_func(x: Sequence[Any], _y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
            regions = []
            for sample in x:
                val = (sample[0] if len(sample) > 0 else 0) if hasattr(sample, "__len__") else float(sample)
                if val < 0.2:
                    regions.append(10)
                else:
                    regions.append(20)
            return np.array(regions, dtype=int)

        regressor = MondrianConformalRegressor(
            model=mock_regression_model,
            score=cast(Any, mock_regression_score),
            region_func=region_func,
        )

        rng = np.random.default_rng(42)
        x_cal = rng.random((10, 5))
        y_cal = rng.random(10)
        regressor.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)

        x_test = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        intervals = regressor.predict(x_test, 0.1)

        assert intervals.shape == (1, 2)


class TestIntegrationAndConsistency:
    def test_region_func_consistency(
        self,
        mock_classification_model: MockModel,
        mock_classification_score: MockClassificationScore,
        simple_region_func: RegionFunc,
    ) -> None:
        classifier = MondrianConformalClassifier(
            model=mock_classification_model,
            score=cast(Any, mock_classification_score),
            region_func=simple_region_func,
        )

        x_cal = [
            [0.1, 0.2],  # region 0
            [0.4, 0.5],  # region 1
            [0.8, 0.9],  # region 2
        ]
        y_cal = [0, 1, 2]

        classifier.calibrate(x_cal, y_cal, 0.1)

        assert 0 in classifier.group_thresholds
        assert 1 in classifier.group_thresholds
        assert 2 in classifier.group_thresholds

    def test_memory_leak_prevention(self) -> None:
        rng = np.random.default_rng(42)
        x_cal = rng.random((1000, 10))
        y_cal = rng.integers(0, 5, 1000)
        x_test = rng.random((100, 10))

        def region_zeros(x: Sequence[Any], _y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
            return np.zeros(len(x), dtype=int)

        classifier = MondrianConformalClassifier(
            model=MockModel(n_classes=5),
            score=cast(Any, MockClassificationScore(n_classes=5)),
            region_func=region_zeros,
        )

        classifier.calibrate(cast(Sequence[Any], x_cal), cast(Sequence[Any], y_cal), 0.1)
        prediction_sets = classifier.predict(cast(Sequence[Any], x_test), 0.1)

        assert prediction_sets.shape == (100, 5)
