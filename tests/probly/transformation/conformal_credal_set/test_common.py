"""Tests for the conformal credal set transformation common code."""

from __future__ import annotations

import pytest


class TestRequireValidations:
    """Internal validation paths in ``_ConformalCredalSetPredictorBase``."""

    def _make_subclass(self):
        """Construct a minimal concrete subclass."""
        from probly.transformation.conformal_credal_set._common import (  # noqa: PLC0415
            _ConformalCredalSetPredictorBase,
        )

        class _Tiny(_ConformalCredalSetPredictorBase):
            pass

        return _Tiny

    def test_require_score_raises_when_score_missing(self) -> None:
        Cls = self._make_subclass()  # noqa: N806

        # Skip __init__ — set attributes directly.
        instance = Cls.__new__(Cls)
        instance.predictor = None
        instance.conformal_quantile = None
        instance.non_conformity_score = None
        with pytest.raises(ValueError, match="No non_conformity_score"):
            instance._require_score()  # noqa: SLF001

    def test_require_calibrated_raises_when_uncalibrated(self) -> None:
        Cls = self._make_subclass()  # noqa: N806

        from probly.conformal_scores import lac_score  # noqa: PLC0415

        instance = Cls.__new__(Cls)
        instance.predictor = None
        instance.conformal_quantile = None
        instance.non_conformity_score = lac_score
        with pytest.raises(ValueError, match="not calibrated"):
            instance._require_calibrated()  # noqa: SLF001

    def test_require_calibrated_returns_quantile_and_score(self) -> None:
        Cls = self._make_subclass()  # noqa: N806
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        instance = Cls.__new__(Cls)
        instance.predictor = None
        instance.conformal_quantile = 0.5
        instance.non_conformity_score = lac_score
        quantile, score = instance._require_calibrated()  # noqa: SLF001
        assert quantile == 0.5
        assert score is lac_score


class TestCalibratedState:
    """``calibrated_state`` returns the (quantile, score) pair from a calibrated predictor."""

    def test_unknown_object_raises(self) -> None:
        from probly.transformation.conformal_credal_set._common import calibrated_state  # noqa: PLC0415

        with pytest.raises(ValueError, match="not a conformal predictor"):
            calibrated_state(object())

    def test_uncalibrated_predictor_raises(self) -> None:
        from probly.conformal_scores import lac_score  # noqa: PLC0415
        from probly.transformation.conformal_credal_set._common import (  # noqa: PLC0415
            _ConformalCredalSetPredictorBase,
            calibrated_state,
        )

        class _Tiny(_ConformalCredalSetPredictorBase):
            pass

        instance = _Tiny.__new__(_Tiny)
        instance.predictor = None
        instance.conformal_quantile = None
        instance.non_conformity_score = lac_score
        with pytest.raises(ValueError, match="not calibrated"):
            calibrated_state(instance)


class TestConformalCredalSetGenerator:
    """The generator dispatch raises for unregistered predictor types."""

    def test_raises_for_unknown_type(self) -> None:
        from probly.conformal_scores import lac_score  # noqa: PLC0415
        from probly.transformation.conformal_credal_set._common import conformal_credal_set_generator  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="No conformal generator"):
            conformal_credal_set_generator(object(), lac_score)
