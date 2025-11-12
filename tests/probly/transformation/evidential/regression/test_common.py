import pytest
import probly.transformation.evidential.regression.common as er_common


def test_evidential_regression_runs():
    class DummyPredictor:
        pass

    dummy = DummyPredictor()
    result = er_common.evidential_regression(dummy)
    assert result is not None


def test_evidential_regression_with_fake_traverse(monkeypatch):
    class DummyPredictor:
        pass

    def fake_traverse(base, composed, init):
        return "ok"

    monkeypatch.setattr(er_common, "traverse", fake_traverse)

    dummy = DummyPredictor()
    result = er_common.evidential_regression(dummy)
    assert result == "ok"


def test_register_calls_traverser(monkeypatch):
    called = {"yes": False}

    class DummyTraverser:
        def register(self, **kwargs):
            called["yes"] = True

    dummy = DummyTraverser()
    monkeypatch.setattr(er_common, "evidential_regression_traverser", dummy)

    er_common.register(cls=object, traverser=object)
    assert called["yes"]
