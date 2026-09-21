"""Backend-agnostic tests for the selective prediction task."""

from __future__ import annotations

import pytest

from probly.evaluation.selective_prediction import selective_prediction


def test_selective_prediction_unregistered_type_raises() -> None:
    with pytest.raises(NotImplementedError, match="selective_prediction"):
        selective_prediction(object(), object())
