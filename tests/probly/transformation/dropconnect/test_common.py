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
    monkeypatch.setattr(
        c.dropconnect_traverser,
        "register",
        fake_register,
        raising=True,
    )

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
    vars_dict = cast(dict[str, object], called["vars"])

    # checks if register call called the right values
    assert called["cls"] is DummyLayer
    assert called["traverser"] is dummy
    assert set(vars_dict.keys()) == {"p"}
    assert vars_dict["p"] is c.P


class FakePredictor:
    pass


def test_dropconnect_uses_p(monkeypatch: pytest.MonkeyPatch) -> None:
    class AlwaysMatchTraverser:
        def match(self, *args: object, **kwargs: object) -> bool:  # noqa: ARG002
            return True

    dummy_traverser = AlwaysMatchTraverser()
    # everytime nn_compose is called, we call our dummy_traverser instead
    monkeypatch.setattr(
        c,
        "nn_compose",
        lambda _: dummy_traverser,
        raising=True,
    )

    # function needs return, so we create a dumy for return_value, but saving all values of the method in "called2"
    called: dict[str, object] = {}
    return_traverse = object()

    def fake_traverse(base: object, composed: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
        assert "init" in kwargs, "init dic must always be passed"
        called["base"] = base
        called["nn_compose"] = composed
        called["init"] = kwargs["init"]
        return return_traverse

    # everytime the travers-method is called, we waant to call our fake_traverse instead
    monkeypatch.setattr(
        c,
        "traverse",
        fake_traverse,
        raising=True,
    )

    base = FakePredictor()
    output = c.dropconnect(base)  # type: ignore[type-var]

    # the fake_traverse always returns the same dummy object
    assert output is return_traverse
    assert called["base"] is base
    assert called["nn_compose"] is dummy_traverser

    init = cast(dict[object, object], called["init"])
    assert init[c.P] == 0.25  # can't check for default since default isn't defined in common -> hardcode 0.25


def test_dropconnect_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapt to code not returning init dict."""
    monkeypatch.setattr(
        c,
        "nn_compose",
        lambda x: x,
        raising=True,
    )

    called: dict[str, object] = {}

    def fake_traverse(_: object, __: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
        assert "init" in kwargs, "init dict must always exist"
        called["init"] = kwargs["init"]
        return object()

    monkeypatch.setattr(
        c,
        "traverse",
        fake_traverse,
        raising=True,
    )

    base = FakePredictor()
    _ = c.dropconnect(
        base,
        p=0.31,
    )  # type: ignore[type-var]

    assert "init" in called
    init = cast(dict[object, object], called["init"])

    assert init[c.P] == 0.31
