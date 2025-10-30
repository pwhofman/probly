from __future__ import annotations

import pytest

from probly.transformation.bayesian import common as c


def test_global_variables() -> None:
    """Check defaults and registration of global variables."""
    assert c.USE_BASE_WEIGHTS.default is False
    assert c.POSTERIOR_STD.default == 0.05
    assert c.PRIOR_MEAN.default == 0.0
    assert c.PRIOR_STD.default == 1.0


class DummyTraverser:
    """Simple placeholder for traverser tests."""


def test_register_calls_bayesian_traverser_register(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify register() correctly forwards variables to bayesian_traverser."""
    called: dict[str, object] = {}

    def fake_register(*, cls: type, traverser: object, variables: dict[str, object]) -> None:
        called["cls"] = cls
        called["traverser"] = traverser
        called["vars"] = variables

    monkeypatch.setattr(c.bayesian_traverser, "register", fake_register, raising=True)

    class DummyLayer:
        pass

    dummy = DummyTraverser()
    c.register(DummyLayer, dummy)

    vars_dict = called["vars"] if isinstance(called["vars"], dict) else {}
    assert called["cls"] is DummyLayer
    assert called["traverser"] is dummy
    assert set(vars_dict.keys()) == {"use_base_weights", "posterior_std", "prior_mean", "prior_std"}
    assert vars_dict["use_base_weights"] is c.USE_BASE_WEIGHTS
    assert vars_dict["posterior_std"] is c.POSTERIOR_STD
    assert vars_dict["prior_mean"] is c.PRIOR_MEAN
    assert vars_dict["prior_std"] is c.PRIOR_STD


class FakePredictor:
    """Dummy predictor for testing."""


def test_bayesian_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that bayesian() uses correct default values."""
    composed_dummy = object()

    def fake_compose(arg: object) -> object:
        assert arg is c.bayesian_traverser
        return composed_dummy

    monkeypatch.setattr("probly.traverse_nn.nn_compose", fake_compose, raising=True)

    called2: dict[str, object] = {}
    return_traverse = object()

    def fake_traverse(base: object, composed: object, init: dict[object, object]) -> object:
        called2["base"] = base
        called2["nn_compose"] = composed
        called2["init"] = init
        return return_traverse

    monkeypatch.setattr("pytraverse.traverse", fake_traverse, raising=True)

    base = FakePredictor()
    output = c.bayesian(base)

    assert output is return_traverse
    assert called2["base"] is base
    assert called2["nn_compose"] is composed_dummy

    init = called2["init"]
    assert isinstance(init, dict)
    assert init[c.USE_BASE_WEIGHTS] is False
    assert init[c.POSTERIOR_STD] == 0.05
    assert init[c.PRIOR_MEAN] == 0.0
    assert init[c.PRIOR_STD] == 1.0
    assert init[c.CLONE] is True


def test_bayesian_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that bayesian() correctly overrides defaults."""
    monkeypatch.setattr("probly.traverse_nn.nn_compose", lambda x: x, raising=True)
    called3: dict[str, object] = {}

    def fake_traverse(_: object, __: object, init: dict[object, object]) -> object:
        called3["init"] = init
        return object()

    monkeypatch.setattr("pytraverse.traverse", fake_traverse, raising=True)

    base = FakePredictor()
    _ = c.bayesian(
        base,
        use_base_weights=True,
        posterior_std=0.4,
        prior_mean=0.5,
        prior_std=1.9,
    )

    init = called3["init"]
    assert isinstance(init, dict)
    assert init[c.USE_BASE_WEIGHTS] is True
    assert init[c.POSTERIOR_STD] == 0.4
    assert init[c.PRIOR_MEAN] == 0.5
    assert init[c.PRIOR_STD] == 1.9
    assert init[c.CLONE] is True
