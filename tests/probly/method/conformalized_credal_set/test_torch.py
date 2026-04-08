"""Tests for TorchConformalizaedCredalSetPredictor."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch as th  # noqa: E402
from torch import nn  # noqa: E402

from probly.method.conformalized_credal_set import conformalized_credal_set  # noqa: E402
from probly.method.conformalized_credal_set.torch import (  # noqa: E402
    TorchConformalizaedCredalSetPredictor,
)
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet  # noqa: E402

NUM_CLASSES = 4
NUM_FEATURES = 8
NUM_CAL = 30
NUM_TEST = 10


@pytest.fixture
def simple_model() -> nn.Module:
    """Return a small softmax classifier."""
    return nn.Sequential(
        nn.Linear(NUM_FEATURES, NUM_CLASSES),
        nn.Softmax(dim=1),
    )


@pytest.fixture
def cal_data() -> tuple[th.Tensor, th.Tensor]:
    """Return synthetic calibration data."""
    th.manual_seed(0)
    x = th.randn(NUM_CAL, NUM_FEATURES)
    y = th.randint(0, NUM_CLASSES, (NUM_CAL,))
    return x, y


@pytest.fixture
def test_data() -> th.Tensor:
    """Return synthetic test inputs."""
    th.manual_seed(1)
    return th.randn(NUM_TEST, NUM_FEATURES)


class TestFactoryFunction:
    """Tests for the conformalized_credal_set factory."""

    def test_returns_torch_predictor(self, simple_model: nn.Module) -> None:
        """Factory should return a TorchConformalizaedCredalSetPredictor for nn.Module."""
        predictor = conformalized_credal_set(simple_model)
        assert isinstance(predictor, TorchConformalizaedCredalSetPredictor)

    def test_predictor_wraps_model(self, simple_model: nn.Module) -> None:
        """Returned predictor should reference the original model."""
        predictor = conformalized_credal_set(simple_model)
        assert predictor.model is simple_model

    def test_not_calibrated_initially(self, simple_model: nn.Module) -> None:
        """Predictor must start in uncalibrated state."""
        predictor = conformalized_credal_set(simple_model)
        assert not predictor.is_calibrated
        assert predictor.threshold is None

    def test_unsupported_type_raises(self) -> None:
        """Factory should raise NotImplementedError for unsupported model types."""
        with pytest.raises(NotImplementedError):
            conformalized_credal_set("not_a_model")  # type: ignore[arg-type]


class TestCalibration:
    """Tests for the calibrate method."""

    def test_calibrate_sets_threshold(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
    ) -> None:
        """After calibration the threshold should be a finite float."""
        x_cal, y_cal = cal_data
        predictor = conformalized_credal_set(simple_model)
        threshold = predictor.calibrate(x_cal, y_cal, alpha=0.1)

        assert predictor.is_calibrated
        assert predictor.threshold is not None
        assert isinstance(threshold, float)
        assert 0.0 <= threshold

    def test_calibrate_returns_threshold(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
    ) -> None:
        """Return value of calibrate should equal the stored threshold."""
        x_cal, y_cal = cal_data
        predictor = conformalized_credal_set(simple_model)
        returned = predictor.calibrate(x_cal, y_cal, alpha=0.1)
        assert returned == predictor.threshold

    def test_higher_alpha_gives_lower_threshold(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
    ) -> None:
        """A higher alpha (less coverage) should yield a lower or equal threshold."""
        x_cal, y_cal = cal_data
        p1 = conformalized_credal_set(simple_model)
        p2 = conformalized_credal_set(simple_model)

        t_low_alpha = p1.calibrate(x_cal, y_cal, alpha=0.05)
        t_high_alpha = p2.calibrate(x_cal, y_cal, alpha=0.5)

        assert t_low_alpha >= t_high_alpha

    def test_calibrate_accepts_numpy_labels(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
    ) -> None:
        """calibrate should accept numpy label arrays."""
        import numpy as np

        x_cal, y_cal = cal_data
        predictor = conformalized_credal_set(simple_model)
        predictor.calibrate(x_cal, y_cal.numpy().astype(np.int64), alpha=0.1)
        assert predictor.is_calibrated


class TestPredict:
    """Tests for the predict method."""

    def test_predict_before_calibration_raises(
        self,
        simple_model: nn.Module,
        test_data: th.Tensor,
    ) -> None:
        """predict should raise RuntimeError when called before calibration."""
        predictor = conformalized_credal_set(simple_model)
        with pytest.raises(RuntimeError, match="calibrated"):
            predictor.predict(test_data)

    def test_predict_returns_credal_set(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
        test_data: th.Tensor,
    ) -> None:
        """predict should return a TorchProbabilityIntervalsCredalSet."""
        x_cal, y_cal = cal_data
        predictor = conformalized_credal_set(simple_model)
        predictor.calibrate(x_cal, y_cal, alpha=0.1)

        result = predictor.predict(test_data)
        assert isinstance(result, TorchProbabilityIntervalsCredalSet)

    def test_credal_set_shape(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
        test_data: th.Tensor,
    ) -> None:
        """The credal set should have shape (n_test, n_classes)."""
        x_cal, y_cal = cal_data
        predictor = conformalized_credal_set(simple_model)
        predictor.calibrate(x_cal, y_cal, alpha=0.1)

        result = predictor.predict(test_data)
        assert result.lower_bounds.shape == (NUM_TEST, NUM_CLASSES)
        assert result.upper_bounds.shape == (NUM_TEST, NUM_CLASSES)

    def test_bounds_valid(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
        test_data: th.Tensor,
    ) -> None:
        """Lower bounds should be >= 0, upper bounds <= 1, lower <= upper."""
        x_cal, y_cal = cal_data
        predictor = conformalized_credal_set(simple_model)
        predictor.calibrate(x_cal, y_cal, alpha=0.1)

        result = predictor.predict(test_data)
        assert th.all(result.lower_bounds >= 0.0)
        assert th.all(result.upper_bounds <= 1.0)
        assert th.all(result.lower_bounds <= result.upper_bounds)

    def test_call_alias(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
        test_data: th.Tensor,
    ) -> None:
        """__call__ should be equivalent to predict."""
        x_cal, y_cal = cal_data
        predictor = conformalized_credal_set(simple_model)
        predictor.calibrate(x_cal, y_cal, alpha=0.1)

        via_predict = predictor.predict(test_data)
        via_call = predictor(test_data)

        assert th.allclose(via_predict.lower_bounds, via_call.lower_bounds)
        assert th.allclose(via_predict.upper_bounds, via_call.upper_bounds)

    def test_larger_alpha_wider_credal_set(
        self,
        simple_model: nn.Module,
        cal_data: tuple[th.Tensor, th.Tensor],
        test_data: th.Tensor,
    ) -> None:
        """More conservative (smaller alpha) should produce wider or equal intervals."""
        x_cal, y_cal = cal_data

        p_conservative = conformalized_credal_set(simple_model)
        p_liberal = conformalized_credal_set(simple_model)

        p_conservative.calibrate(x_cal, y_cal, alpha=0.05)
        p_liberal.calibrate(x_cal, y_cal, alpha=0.5)

        cs_conservative = p_conservative.predict(test_data)
        cs_liberal = p_liberal.predict(test_data)

        width_conservative = cs_conservative.width().mean()
        width_liberal = cs_liberal.width().mean()

        assert width_conservative >= width_liberal

    def test_str_representation(self, simple_model: nn.Module) -> None:
        """__str__ should mention the model class and calibration status."""
        predictor = conformalized_credal_set(simple_model)
        s = str(predictor)
        assert "not calibrated" in s
        assert "Sequential" in s
