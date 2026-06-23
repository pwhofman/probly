"""Backend-agnostic tests for efficient_credal_prediction/_common.py.

Covers the dispatch-fallback errors, the alpha validator, and the
``EfficientCredalRepresenter`` precondition check.
"""

from __future__ import annotations

import pytest

from probly.method.efficient_credal_prediction._common import (
    _validate_alpha,
    compute_efficient_credal_bounds,
    compute_efficient_credal_prediction_bounds,
    efficient_credal_prediction_generator,
)


class TestEfficientCredalPredictionGenerator:
    """Default branch of the predictor generator (lines 44-45)."""

    def test_unregistered_predictor_type_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="No efficient credal prediction generator"):
            efficient_credal_prediction_generator("not-a-predictor")  # type: ignore[arg-type]


class TestValidateAlpha:
    """``_validate_alpha`` enforces the ``[0, 1]`` interval (lines 70-71)."""

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            _validate_alpha(-0.1)

    def test_alpha_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            _validate_alpha(1.5)

    def test_boundary_values_are_accepted(self) -> None:
        # Boundary values must not raise.
        _validate_alpha(0.0)
        _validate_alpha(1.0)


class TestComputeBoundsDispatch:
    """Default branches of the dispatch fallbacks (lines 86-87, 109-110)."""

    def test_unregistered_array_type_raises_for_prediction_bounds(self) -> None:
        with pytest.raises(NotImplementedError, match="No credal bounds computation"):
            compute_efficient_credal_prediction_bounds(
                "not-an-array",  # type: ignore[arg-type]
                "not-an-array",  # type: ignore[arg-type]
                num_classes=2,
                alpha=0.5,
            )

    def test_unregistered_array_type_raises_for_credal_bounds(self) -> None:
        with pytest.raises(NotImplementedError, match="No compute_efficient_credal_bounds implementation"):
            compute_efficient_credal_bounds("not-an-array", "lower", "upper")  # type: ignore[arg-type]


class TestEfficientCredalRepresenterRequiresBounds:
    """The representer must reject uninitialized bounds (lines 135-141)."""

    def test_represent_without_bounds_raises(self) -> None:
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")
        from probly.method.efficient_credal_prediction import (  # noqa: PLC0415
            EfficientCredalRepresenter,
            efficient_credal_prediction,
        )

        predictor = efficient_credal_prediction(nn.Linear(2, 2), predictor_type="logit_classifier")
        # Both lower and upper start as ``None``; calling ``represent`` must raise.
        rep = EfficientCredalRepresenter(predictor)

        with pytest.raises(RuntimeError, match="uninitialized bounds"):
            rep.represent(torch.ones(1, 2))
