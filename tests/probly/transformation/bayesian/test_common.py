from __future__ import annotations

from typing import Any, cast

import pytest

from probly.transformation.bayesian import common as c


# check if the global variable defaults are correct
def test_global_variables() -> None:
    assert c.USE_BASE_WEIGHTS.default is False
    assert c.POSTERIOR_STD.default == 0.05
    assert c.PRIOR_MEAN.default == 0.0
    assert c.PRIOR_STD.default == 1.0


# fake traverser
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

    # everytime the register method of our bayesian_traverser is called, we monkeypatch
    # it to replace it with our fake_register
    monkeypatch.setattr(
        c.bayesian_traverser,
        "register",
        fake_register,
        raising=True,
    )

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
    assert set(vars_dict.keys()) == {
        "use_base_weights",
        "posterior_std",
        "prior_mean",
        "prior_std",
    }
    assert vars_dict["use_base_weights"] is c.USE_BASE_WEIGHTS
    assert vars_dict["posterior_std"] is c.POSTERIOR_STD
    assert vars_dict["prior_mean"] is c.PRIOR_MEAN
    assert vars_dict["prior_std"] is c.PRIOR_STD


class FakePredictor:
    """Dummy predictor for testing."""


def test_bayesian_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch traverse call so output always matches expected dummy."""

    class AlwaysMatchTraverser:
        def match(self, *args: object, **kwargs: object) -> bool:  # noqa: ARG002
            return True

    dummy_traverser = AlwaysMatchTraverser()
    # everytime nn_compose is called, we call our fake_compose instead
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
    output = c.bayesian(base)  # type: ignore[type-var]

    # the fake_traverse always returns the same dummy object
    assert output is return_traverse
    assert called["base"] is base
    assert called["nn_compose"] is dummy_traverser

    init = cast(dict[object, object], called["init"])
    assert init[c.USE_BASE_WEIGHTS] == c.USE_BASE_WEIGHTS.default
    assert init[c.POSTERIOR_STD] == c.POSTERIOR_STD.default
    assert init[c.PRIOR_MEAN] == c.PRIOR_MEAN.default
    assert init[c.PRIOR_STD] == c.PRIOR_STD.default


def test_bayesian_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check override values are forwarded exactly."""
    # nn_compose soll einfach den Traverser zurückgeben
    monkeypatch.setattr(
        c,
        "nn_compose",
        lambda t: t,
        raising=True,
    )

    called: dict[str, object] = {}

    # Fake traverse, fängt init-Dict ab
    def fake_traverse(base: object, composed: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
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
    _ = c.bayesian(
        base,
        use_base_weights=True,
        posterior_std=0.4,
        prior_mean=0.5,
        prior_std=1.8,
    )  # type: ignore[type-var]

    # init dict muss existieren
    assert "init" in called
    init = cast(dict[object, object], called["init"])

    # EXAKT erwartete Werte
    assert init[c.USE_BASE_WEIGHTS] is True
    assert init[c.POSTERIOR_STD] == 0.4
    assert init[c.PRIOR_MEAN] == 0.5
    assert init[c.PRIOR_STD] == 1.8
