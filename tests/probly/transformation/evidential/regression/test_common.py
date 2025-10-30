"""Tests for the probly.transformation.evidential.regression module (shared, no torch)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

pytest_plugins = [
    "tests.probly.fixtures.common",
    "tests.probly.fixtures.torch_models",
    "tests.probly.fixtures.flax_models",
]

import types
import pytest

# import both the regmod so you can monkeypatch and the functions from the main method 
import probly.transformation.evidential.regression.common as regmod
from probly.transformation.evidential.regression.common import (
    evidential_regression,
    register,
   REPLACED_LAST_LINEAR,
)
from pytraverse import CLONE, TRAVERSE_REVERSED

# recorded -> container for the outputs 
def test_evidential_regression_calls_traverse_with_reverse_and_clone(monkeypatch):
    """It should traverse the base predictor in reverse order and clone it."""
    recorded = {}

    def fake_traverse(base, composed, init):
        # captures the call
        recorded["base"]     = base
        recorded["composed"] = composed
        recorded["init"]     = init
        return "TRAVERSE_RESULT"

    def fake_nn_compose(trav):
        # done so we can verify the exact traverser object the code passed 
        return ("COMPOSED", trav)

    # patch inside the module under test (important: patch regmod.*, not the library names)
    monkeypatch.setattr(regmod, "traverse", fake_traverse)
    monkeypatch.setattr(regmod, "nn_compose", fake_nn_compose)

    base_model = "BASE_PREDICTOR"
    result = evidential_regression(base_model)

    # returns whatever traverse returns
    assert result == "TRAVERSE_RESULT"

    # base object was passed through
    assert recorded["base"] == base_model

    # the composed function contains the module's traverser
    composed, threaded_trav = recorded["composed"]
    assert composed        == "COMPOSED"
    assert threaded_trav is regmod.evidential_regression_traverser

    # flags are set as required
    init = recorded["init"]
    assert init[TRAVERSE_REVERSED] is True
    assert init[CLONE] is True


def test_register_forwards_and_uses_global_skip_flag(monkeypatch):
    """register(...) should wire skip_if to the REPLACED_LAST_LINEAR GlobalVariable."""
    call = {}

    def fake_register(*, cls, traverser, skip_if):
        call["cls"]       = cls
        call["traverser"] = traverser
        call["skip_if"]   = skip_if

    # monkeypatch the lazydispatch_traverser's register method
    monkeypatch.setattr(regmod.evidential_regression_traverser, "register", fake_register)

    class DummyLayer:
        pass

    def dummy_traverser(x, traverse):
        return x

    # act
    register(DummyLayer, dummy_traverser)

    # forwarded args
    assert call["cls"] is DummyLayer
    assert call["traverser"] is dummy_traverser

    # the wired skip_if must consult the REPLACED_LAST_LINEAR global var in traversal state
    skip_if = call["skip_if"]
    state_false = {REPLACED_LAST_LINEAR: False}
    state_true = {REPLACED_LAST_LINEAR: True}
    assert skip_if(state_false) is False  # not replaced yet -> don't skip
    assert skip_if(state_true) is True    # already replaced -> skip


def test_idempotent_behavior_contract_via_skip(monkeypatch):
    """
    Check the intended 'replace only the last one' contract at the shared layer:
    if the state marks REPLACED_LAST_LINEAR=True, the registered traverser should be skipped.
    We validate this by capturing the 'skip_if' that register() passes down.
    """
    captured = {}

    def fake_register(*, cls, traverser, skip_if):
        captured["skip_if"] = skip_if

    monkeypatch.setattr(regmod.evidential_regression_traverser, "register", fake_register)

    class Foo:
        pass

    def noop(x, traverse):
        return x

    register(Foo, noop)
    skip_if = captured["skip_if"]

    # When replacement already happened, further matches must be skipped:
    assert skip_if({REPLACED_LAST_LINEAR: True}) is True
    # When it hasn't, they must proceed:
    assert skip_if({REPLACED_LAST_LINEAR: False}) is False
