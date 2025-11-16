from __future__ import annotations

import importlib

import pytest


def _state_with(flag, value):
    class _S:
        def __getitem__(self, k):
            if k is flag:
                return value
            raise KeyError

    return _S()


def test_register_attaches_skip_if_and_uses_global_flag():
    from probly.transformation.evidential.regression import common

    importlib.reload(common)

    captured = {}

    def _capture_register(**kwargs):
        captured["kwargs"] = kwargs

    original_register = common.evidential_regression_traverser.register
    try:
        common.evidential_regression_traverser.register = _capture_register  # type: ignore[attr-defined]

        class DummyCls: ...

        dummy_traverser = object()
        common.register(DummyCls, dummy_traverser)
    finally:
        common.evidential_regression_traverser.register = original_register  # type: ignore[attr-defined]

    kwargs = captured["kwargs"]
    assert kwargs["cls"] is DummyCls
    assert kwargs["traverser"] is dummy_traverser

    skip_if = kwargs["skip_if"]
    flag = common.REPLACED_LAST_LINEAR
    assert skip_if(_state_with(flag, True)) is True
    assert skip_if(_state_with(flag, False)) is False
    with pytest.raises(KeyError):
        skip_if(_state_with(object(), True))


def test_evidential_regression_calls_traverse_with_expected_init_and_compose():
    from probly.transformation.evidential.regression import common

    importlib.reload(common)

    calls = {}

    def fake_nn_compose(arg):
        calls["nn_compose_arg"] = arg
        return "COMPOSED"

    def fake_traverse(base, composed, init):
        calls["traverse_args"] = (base, composed, init)
        return "RESULT"

    orig_nn_compose = common.nn_compose
    orig_traverse = common.traverse
    try:
        common.nn_compose = fake_nn_compose
        common.traverse = fake_traverse

        class BasePredictor: ...

        base = BasePredictor()
        result = common.evidential_regression(base)
    finally:
        common.nn_compose = orig_nn_compose
        common.traverse = orig_traverse

    assert calls["nn_compose_arg"] is common.evidential_regression_traverser
    base_arg, composed_arg, init_arg = calls["traverse_args"]
    assert base_arg is base
    assert composed_arg == "COMPOSED"
    assert init_arg.get(common.TRAVERSE_REVERSED) is True
    assert init_arg.get(common.CLONE) is True
    assert result == "RESULT"


@pytest.mark.parametrize("flag_value", [True, False])
def test_skip_logic_via_flag_end_to_end(flag_value):
    from probly.transformation.evidential.regression import common

    importlib.reload(common)

    captured = {}

    def _capture_register(**kwargs):
        captured["skip_if"] = kwargs["skip_if"]

    original_register = common.evidential_regression_traverser.register
    try:
        common.evidential_regression_traverser.register = _capture_register  # type: ignore[attr-defined]
        common.register(object, object())
    finally:
        common.evidential_regression_traverser.register = original_register  # type: ignore[attr-defined]

    skip_if = captured["skip_if"]
    flag = common.REPLACED_LAST_LINEAR
    assert skip_if(_state_with(flag, flag_value)) is flag_value
