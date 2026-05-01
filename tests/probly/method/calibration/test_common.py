"""Common tests for calibration method factories."""

from __future__ import annotations

import pytest

from probly.method.calibration import vector_scaling
from probly.predictor import LogitClassifier, Predictor
from probly.transformation.calibration._common import CalibrationMethodConfig, calibration_generator


def test_unregistered_calibration_generator_raises(dummy_predictor: Predictor) -> None:
    """No calibration generator is registered for unsupported predictor types."""
    config = CalibrationMethodConfig(method="temperature", vector_scale=False, use_bias=False)
    with pytest.raises(
        NotImplementedError, match=f"No calibration generator is registered for type {type(dummy_predictor)}"
    ):
        calibration_generator(dummy_predictor, config)


def test_vector_scaling_rejects_invalid_num_classes(dummy_predictor: Predictor) -> None:
    """Vector scaling validates class-count hints at construction time."""
    LogitClassifier.register_instance(dummy_predictor)
    msg = "vector scaling expects num_classes > 1"
    with pytest.raises(ValueError, match=msg):
        vector_scaling(dummy_predictor, num_classes=1)
