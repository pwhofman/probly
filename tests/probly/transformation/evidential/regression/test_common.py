import sys
import types
import importlib
import pytest

class FakeGlobalVariable:
    def __init__(self, name, description, default=False):
        self.name = name
        self.description = description
        self.default = default
    def __class_getitem__(cls, item):
        return cls

class FakeTraverser:
    def __init__(self, name):
        self.name = name
        self.register_calls = []
    def register(self, **kwargs):
        self.register_calls.append(kwargs)

class FakeLazyDispatch:
    def __getitem__(self, item):
        def creator(name):
            return FakeTraverser(name=name)
        return creator

def install_fakes():
    probly = types.ModuleType("probly")
    traverse_nn = types.ModuleType("probly.traverse_nn")
    pytraverse = types.ModuleType("pytraverse")

    def nn_compose(x):
        return ("COMPOSED", x)
    traverse_nn.nn_compose = nn_compose

    CLONE = object()
    TRAVERSE_REVERSED = object()

    def traverse(base, composed, init):
        traverse.last_call = (base, composed, init)
        return "TRANSFORMED"

    pytraverse.CLONE = CLONE
    pytraverse.TRAVERSE_REVERSED = TRAVERSE_REVERSED
    pytraverse.GlobalVariable = FakeGlobalVariable
    pytraverse.lazydispatch_traverser = FakeLazyDispatch()
    pytraverse.traverse = traverse

    sys.modules["probly"] = probly
    sys.modules["probly.traverse_nn"] = traverse_nn
    sys.modules["pytraverse"] = pytraverse

@pytest.fixture(autouse=True)
def clean_imports():
    victims = [k for k in list(sys.modules) if k.startswith("probly.transformation.evidential.regression.common")]
    for k in victims:
        sys.modules.pop(k, None)
    yield
    for k in victims:
        sys.modules.pop(k, None)

def test_register_and_skip_if_behaviour():
    install_fakes()
    mod = importlib.import_module("probly.transformation.evidential.regression.common")
    assert isinstance(mod.REPLACED_LAST_LINEAR, FakeGlobalVariable)
    assert mod.REPLACED_LAST_LINEAR.default is False
    assert mod.evidential_regression_traverser.name == "evidential_regression_traverser"
    dummy_cls = object()
    dummy_trav = object()
    mod.register(dummy_cls, dummy_trav)
    assert mod.evidential_regression_traverser.register_calls
    call = mod.evidential_regression_traverser.register_calls[-1]
    assert call["cls"] is dummy_cls
    assert call["traverser"] is dummy_trav
    skip_if = call["skip_if"]
    assert skip_if({mod.REPLACED_LAST_LINEAR: True}) is True
    assert skip_if({mod.REPLACED_LAST_LINEAR: False}) is False

def test_evidential_regression_traverse_called_and_returns_transformed():
    install_fakes()
    mod = importlib.import_module("probly.transformation.evidential.regression.common")
    base = object()
    result = mod.evidential_regression(base)
    assert result == "TRANSFORMED"
    base_arg, composed_arg, init_arg = sys.modules["pytraverse"].traverse.last_call
    assert base_arg is base
    tag, traverser = composed_arg
    assert tag == "COMPOSED"
    assert isinstance(traverser, FakeTraverser)
    assert traverser.name == "evidential_regression_traverser"
    assert init_arg[sys.modules["pytraverse"].TRAVERSE_REVERSED] is True
    assert init_arg[sys.modules["pytraverse"].CLONE] is True
