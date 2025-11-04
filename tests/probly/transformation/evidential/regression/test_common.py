"""Tests for the probly.transformation.evidential.regression module (shared, no torch)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import matplotlib as mpl
import pytest

import probly.transformation.evidential.regression.common as regmod
from probly.transformation.evidential.regression.common import (
    REPLACED_LAST_LINEAR,
    evidential_regression,
    register,
)
from pytraverse import CLONE, TRAVERSE_REVERSED, State

mpl.use("Agg")

pytest_plugins = [
    "tests.probly.fixtures.common",
    "tests.probly.fixtures.torch_models",
    "tests.probly.fixtures.flax_models",
]


def test_evidential_regression_calls_traverse_with_reverse_and_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It should traverse the base predictor in reverse order and clone it."""
    recorded: dict[str, object] = {}

    def fake_traverse(base: object, composed: object, init: State[object]) -> str:
        recorded["base"] = base
        recorded["composed"] = composed
        recorded["init"] = init
        return "TRAVERSE_RESULT"

    def fake_nn_compose(trav: object) -> tuple[str, object]:
        return ("COMPOSED", trav)

    monkeypatch.setattr(regmod, "traverse", fake_traverse, raising=True)
    monkeypatch.setattr(regmod, "nn_compose", fake_nn_compose, raising=True)

    # fake predictor
    base_model = cast(Any, "BASE_PREDICTOR")
    result = evidential_regression(base_model)

    assert result == "TRAVERSE_RESULT"
    assert recorded["base"] == base_model

    composed, threaded_trav = cast(tuple[str, object], recorded["composed"])
    assert composed == "COMPOSED"
    assert threaded_trav is regmod.evidential_regression_traverser

    init = cast(State[object], recorded["init"])
    assert init[TRAVERSE_REVERSED] is True
    assert init[CLONE] is True


def test_register_forwards_and_uses_global_skip_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """register(...) should wire skip_if to the REPLACED_LAST_LINEAR GlobalVariable."""
    call_info: dict[str, object] = {}

    def fake_register(
        *,
        cls: type[object],
        traverser: Callable[[object, Callable[..., object]], object],
        skip_if: Callable[[State[object]], bool],
    ) -> None:
        call_info["cls"] = cls
        call_info["traverser"] = traverser
        call_info["skip_if"] = skip_if

    monkeypatch.setattr(
        regmod.evidential_regression_traverser,
        "register",
        fake_register,
        raising=True,
    )

    class DummyLayer:
        """Dummy layer class for registration testing."""

    def dummy_traverser(x: object, _traverse: Callable[..., object]) -> object:
        return x

    register(DummyLayer, dummy_traverser)

    assert call_info["cls"] is DummyLayer
    assert call_info["traverser"] is dummy_traverser

    skip_if = cast(Callable[[State[object]], bool], call_info["skip_if"])
    state_false = cast(State[object], {REPLACED_LAST_LINEAR: False})
    state_true = cast(State[object], {REPLACED_LAST_LINEAR: True})
    assert skip_if(state_false) is False
    assert skip_if(state_true) is True


def test_idempotent_behavior_contract_via_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure only the last linear layer is replaced by consulting the global flag."""
    captured: dict[str, object] = {}

    def fake_register(
        *,
        cls: type[object],  # noqa: ARG001
        traverser: Callable[[object, Callable[..., object]], object],  # noqa: ARG001
        skip_if: Callable[[State[object]], bool],
    ) -> None:
        captured["skip_if"] = skip_if

    monkeypatch.setattr(
        regmod.evidential_regression_traverser,
        "register",
        fake_register,
        raising=True,
    )

    class Foo:
        """Dummy model."""

    def noop(x: object, _traverse: Callable[..., object]) -> object:
        return x

    register(Foo, noop)
    skip_if = cast(Callable[[State[object]], bool], captured["skip_if"])

    assert skip_if(cast(State[object], {REPLACED_LAST_LINEAR: True})) is True
    assert skip_if(cast(State[object], {REPLACED_LAST_LINEAR: False})) is False
