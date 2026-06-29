"""Flax serialization behavior for probly conformal wrappers."""

from __future__ import annotations

from typing import Any

import pytest

from flextype import registry_pickle
from probly.method.conformal import LACConformalSetPredictor, conformal_lac
from probly.predictor import LogitClassifier

pytest.importorskip("jax")
pytest.importorskip("flax")
from flax import nnx
import jax.numpy as jnp


def _state_scalar(state: nnx.State, key: str) -> Any:  # noqa: ANN401
    """Return a scalar state leaf from an nnx state mapping."""
    value = state[key]
    if isinstance(value, nnx.Variable):
        return value.value
    return value


def _make_flax_logits_model(seed: int) -> nnx.Module:
    """Build a small flax model that emits logits."""
    rngs = nnx.Rngs(seed)
    return nnx.Sequential(
        nnx.Linear(2, 2, rngs=rngs),
        nnx.Linear(2, 2, rngs=rngs),
        nnx.Linear(2, 2, rngs=rngs),
    )


def test_flax_conformal_quantile_is_persisted_in_nnx_state() -> None:
    """Conformal quantile should be included in nnx state for flax wrappers."""
    model = conformal_lac(_make_flax_logits_model(0), predictor_type=LogitClassifier)

    assert model.conformal_quantile is None
    model.conformal_quantile = 0.42

    state = nnx.state(model)
    quantile_state = _state_scalar(state, "_conformal_quantile")

    assert "_conformal_quantile" in state
    assert not bool(jnp.isnan(quantile_state))
    assert float(quantile_state) == pytest.approx(0.42)


def test_flax_conformal_quantile_restores_via_nnx_update() -> None:
    """Updating flax wrapper state should restore conformal quantile."""
    source = conformal_lac(_make_flax_logits_model(1), predictor_type=LogitClassifier)
    source.conformal_quantile = 0.37

    target = conformal_lac(_make_flax_logits_model(2), predictor_type=LogitClassifier)
    assert target.conformal_quantile is None

    nnx.update(target, nnx.state(source))

    assert target.conformal_quantile == pytest.approx(0.37)


def test_flax_conformal_quantile_none_roundtrips_as_nan_state() -> None:
    """Uncalibrated flax wrappers should keep conformal quantile as None after state restore."""
    source = conformal_lac(_make_flax_logits_model(3), predictor_type=LogitClassifier)
    assert source.conformal_quantile is None

    state = nnx.state(source)
    quantile_state = _state_scalar(state, "_conformal_quantile")
    assert "_conformal_quantile" in state
    assert bool(jnp.isnan(quantile_state))

    target = conformal_lac(_make_flax_logits_model(4), predictor_type=LogitClassifier)
    target.conformal_quantile = 0.99
    assert target.conformal_quantile == pytest.approx(0.99)

    nnx.update(target, state)

    assert target.conformal_quantile is None


def test_flax_conformal_wrapper_roundtrip_with_registry_pickle_preserves_quantile() -> None:
    """Full-object registry pickle round-trip should preserve flax conformal wrapper type and quantile."""
    wrapped = conformal_lac(_make_flax_logits_model(5), predictor_type=LogitClassifier)
    wrapped.conformal_quantile = 0.61

    restored = registry_pickle.loads(registry_pickle.dumps(wrapped))

    assert type(restored) is type(wrapped)
    assert isinstance(restored, LACConformalSetPredictor)
    assert restored.conformal_quantile == pytest.approx(0.61)
