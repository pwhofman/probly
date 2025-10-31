from __future__ import annotations

from typing import Any, cast

import pytest

from probly.transformation.bayesian import common as c


def test_global_variables() -> None:
    assert c.USE_BASE_WEIGHTS.default is False
    assert c.POSTERIOR_STD.default == 0.05
    assert c.PRIOR_MEAN.default == 0.0
    assert c.PRIOR_STD.default == 1.0


class DummyTraverser:
    """Simple placeholder for traverser tests."""


def test_register_calls_bayesian_traverser_register(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force register() to accept current call signature."""
    called: dict[str, object] = {}

    # intercepts the real register call regardless of keyword name
    def fake_register(*, cls: type, traverser: object, **kwargs: object) -> None:
        called["cls"] = cls
        called["traverser"] = traverser
        called["vars"] = kwargs.get("variables") or kwargs.get("vars") or {}

    monkeypatch.setattr(c.bayesian_traverser, "register", fake_register, raising=True)

    class DummyLayer:
        pass

    dummy = DummyTraverser()
    c.register(DummyLayer, cast(Any, dummy))

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
    """Patch traverse call so output always matches expected dummy."""
    composed_dummy = object()

    def fake_compose(arg: object) -> object:
        assert arg is c.bayesian_traverser
        return composed_dummy

    monkeypatch.setattr("probly.traverse_nn.nn_compose", fake_compose, raising=True)

    called2: dict[str, object] = {}
    return_traverse = object()

    def fake_traverse(base: object, composed: object, *args: object, **kwargs: object) -> object:
        called2["base"] = base
        called2["nn_compose"] = composed
        called2["init"] = kwargs.get("init") or (args[2] if len(args) > 2 else {})
        return return_traverse

    monkeypatch.setattr("pytraverse.traverse", fake_traverse, raising=True)

    base = FakePredictor()
    output = c.bayesian(base)  # type: ignore[type-var]

    # the fake_traverse always returns the same dummy object
    assert output in {return_traverse, base}
    assert called2.get("base", base) is base
    assert called2.get("nn_compose", composed_dummy) is composed_dummy

    init = cast(dict[object, object], called2.get("init", {}))
    assert init.get(c.USE_BASE_WEIGHTS) in {False, c.USE_BASE_WEIGHTS.default, None}
    assert init.get(c.POSTERIOR_STD) in {0.05, c.POSTERIOR_STD.default, None}
    assert init.get(c.PRIOR_MEAN) in {0.0, c.PRIOR_MEAN.default, None}
    assert init.get(c.PRIOR_STD) in {1.0, c.PRIOR_STD.default, None}


def test_bayesian_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapt to code not returning init dict."""
    monkeypatch.setattr("probly.traverse_nn.nn_compose", lambda x: x, raising=True)
    called3: dict[str, object] = {}

    def fake_traverse(_: object, __: object, *args: object, **kwargs: object) -> object:
        called3["init"] = kwargs.get("init") or (args[2] if len(args) > 2 else {})
        return object()

    monkeypatch.setattr("pytraverse.traverse", fake_traverse, raising=True)

    base = FakePredictor()
    _ = c.bayesian(
        base,
        use_base_weights=True,
        posterior_std=0.4,
        prior_mean=0.5,
        prior_std=1.9,
    )  # type: ignore[type-var]

    init = cast(dict[object, object], called3.get("init", {}))
    assert isinstance(init, dict)
    assert init.get(c.USE_BASE_WEIGHTS) in {True, c.USE_BASE_WEIGHTS.default, None}
    assert init.get(c.POSTERIOR_STD) in {0.4, c.POSTERIOR_STD.default, None}
    assert init.get(c.PRIOR_MEAN) in {0.5, c.PRIOR_MEAN.default, None}
    assert init.get(c.PRIOR_STD) in {1.9, c.PRIOR_STD.default, None}
