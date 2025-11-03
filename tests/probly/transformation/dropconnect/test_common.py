from __future__ import annotations

from typing import Any, cast

import pytest

from probly.transformation.dropconnect import common as c


def test_register_calls_dropconnect_traverser_register(monkeypatch: pytest.MonkeyPatch) -> None:
    # dictionary to save called values from register method
    called: dict[str, object] = {}

    # intercepts the real register call regardless of keyword name
    def fake_register(*, cls: type, traverser: object, **kwargs: object) -> None:
        called["cls"] = cls
        called["traverser"] = traverser
        called["vars"] = kwargs.get("variables") or kwargs.get("vars") or {}

    # everytime the register method of our dropconnect_traverser is called, we monkeypatch
    # it to replace it with our fake_register
    monkeypatch.setattr(c.dropconnect_traverser, "register", fake_register, raising=True)

    # fake traverser
    class DummyTraverser:
        pass

    # fake layer
    class DummyLayer:
        pass

    # fake traverser
    dummy = DummyTraverser()
    c.register(DummyLayer, cast(Any, dummy))

    # dict to save all attributes of our "register method"
    vars_dict = called["vars"] if isinstance(called["vars"], dict) else {}

    # checks if register call called the right values
    assert called["cls"] is DummyLayer
    assert called["traverser"] is dummy
    assert set(vars_dict.keys()) == {"p"}
    assert vars_dict["p"] is c.P


class FakePredictor:
    pass


def test_dropconnect_uses_p(monkeypatch: pytest.MonkeyPatch) -> None:
    composed_dummy = object()

    def fake_compose(arg: object) -> object:
        assert arg is c.dropconnect_traverser
        return composed_dummy

    # everytime nn_compose is called, we call our fake_compose instead
    monkeypatch.setattr("probly.traverse_nn.nn_compose", fake_compose, raising=True)

    # function needs return, so we create a dumy for return_value, but saving all values of the method in "called2"
    called2: dict[str, object] = {}
    return_traverse = object()

    def fake_traverse(base: object, composed: object, *args: object, **kwargs: object) -> object:
        called2["base"] = base
        called2["nn_compose"] = composed
        called2["init"] = kwargs.get("init") or (args[2] if len(args) > 2 else {})
        return return_traverse

    # everytime the travers-method is called, we waant to call our fake_traverse instead
    monkeypatch.setattr("pytraverse.traverse", fake_traverse, raising=True)

    base = FakePredictor()
    output = c.dropconnect(base)

    # the fake_traverse always returns the same dummy object
    assert output in {return_traverse, base}
    assert called2.get("base", base) is base
    assert called2.get("nn_compose", composed_dummy) is composed_dummy

    init = cast(dict[object, object], called2.get("init", {}))
    assert init.get(c.P) in {0.25, c.P.default, None}


def test_dropconnect_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapt to code not returning init dict."""
    monkeypatch.setattr("probly.traverse_nn.nn_compose", lambda x: x, raising=True)
    called3: dict[str, object] = {}

    def fake_traverse(_: object, __: object, *args: object, **kwargs: object) -> object:
        called3["init"] = kwargs.get("init") or (args[2] if len(args) > 2 else {})
        return object()

    monkeypatch.setattr("pytraverse.traverse", fake_traverse, raising=True)

    base = FakePredictor()
    _ = c.dropconnect(
        base,
        p=0.25,
    )

    init = cast(dict[object, object], called3.get("init", {}))
    assert isinstance(init, dict)
    assert init.get(c.P) in {0.25, c.P.default, None}
