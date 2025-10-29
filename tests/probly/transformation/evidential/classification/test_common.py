from __future__ import annotations

import pytest

from probly.transformation.evidential import evidential_classification


def test_invalid_base() -> None:
    class DummyPredictor:
        pass

    base = DummyPredictor()
    with pytest.raises(
        NotImplementedError,
        match=f"No evidential classification appender registered for type {type(base)}",
    ):
        evidential_classification(base)  # type: ignore[type-var]
