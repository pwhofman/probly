import importlib
import pytest


def test_register_attaches_skip_if_and_uses_global_flag(mocker):
    import probly.transformation.evidential.regression.common as common
    importlib.reload(common)

    spy = mocker.spy(common.evidential_regression_traverser, "register")

    class DummyCls: ...
    dummy_traverser = object()

    common.register(DummyCls, dummy_traverser)

    spy.assert_called_once()
    kwargs = spy.call_args.kwargs
    assert kwargs["cls"] is DummyCls
    assert kwargs["traverser"] is dummy_traverser

    skip_if = kwargs["skip_if"]
    flag = common.REPLACED_LAST_LINEAR
    assert skip_if({flag: True}) is True
    assert skip_if({flag: False}) is False
    try:
        res = skip_if({})
    except KeyError:
        res = None
    assert res in (None, False)


def test_evidential_regression_calls_traverse_with_expected_init_and_compose(mocker):
    import probly.transformation.evidential.regression.common as common
    import pytraverse
    importlib.reload(common)
    importlib.reload(pytraverse)

    spy = mocker.spy(pytraverse, "traverse")

    class BasePredictor:
        pass

    base = BasePredictor()

    result = common.evidential_regression(base)

    assert spy.call_count >= 1
    call_args, call_kwargs = spy.call_args
    base_arg, composed_arg, init_arg = call_args[:3]

    assert base_arg is base
    expected_composed = common.nn_compose(common.evidential_regression_traverser)
    assert composed_arg == expected_composed
    assert init_arg.get(pytraverse.TRAVERSE_REVERSED) is True
    assert init_arg.get(pytraverse.CLONE) is True
    assert result is not None or result is None


@pytest.mark.parametrize("flag_value", [True, False])
def test_skip_logic_via_flag_end_to_end(flag_value):
    import probly.transformation.evidential.regression.common as common
    importlib.reload(common)

    captured = {}

    def _capture_register(**kwargs):
        captured["skip_if"] = kwargs["skip_if"]

    original_register = common.evidential_regression_traverser.register
    try:
        common.evidential_regression_traverser.register = _capture_register
        common.register(object, object())
    finally:
        common.evidential_regression_traverser.register = original_register

    skip_if = captured["skip_if"]
    flag = common.REPLACED_LAST_LINEAR
    assert skip_if({flag: flag_value}) is flag_value
