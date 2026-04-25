"""Tests for model wrappers."""

from __future__ import annotations

import pytest

from river_uq.models import build_model
from river_uq.streams import build_stream


@pytest.mark.parametrize("kind", ["arf", "deep_ensemble", "mc_dropout"])
def test_wrapper_learns_and_quantifies(kind: str) -> None:
    pytest.importorskip("torch")
    model = build_model(kind, seed=0)
    stream, _ = build_stream("stagger_drift", seed=0, n=200)

    for x, y in stream:
        model.learn_one(x, y)

    last_x, _ = next(iter(build_stream("stagger_drift", seed=0, n=1)[0]))
    pred = model.predict_one(last_x)
    assert isinstance(pred, int)

    decomp = model.epistemic_decomposition(last_x)
    assert decomp.epistemic >= 0.0
    assert decomp.aleatoric >= 0.0
    assert decomp.total >= 0.0


def test_arf_native_drift_count() -> None:
    """ARF wrapper exposes river's native drift counter for the baseline."""
    model = build_model("arf", seed=0)
    stream, _ = build_stream("stagger_drift", seed=0, n=200)
    for x, y in stream:
        model.learn_one(x, y)
    assert hasattr(model, "n_drifts_detected")
    assert model.n_drifts_detected >= 0


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_model("nope", seed=0)
