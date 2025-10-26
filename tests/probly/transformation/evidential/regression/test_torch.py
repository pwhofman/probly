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

class FakeLinear:
    def __init__(self, in_features, out_features, bias=True, device="cpu"):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = types.SimpleNamespace(device=device)
        self.bias = object() if bias else None

class FakeNormalInverseGammaLinear:
    def __init__(self, in_features, out_features, device=None, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.bias = bias

@pytest.fixture(autouse=True)
def clean_imports():
    victims = [k for k in list(sys.modules) if k.startswith("probly.transformation.evidential.regression")]
    victims += ["torch", "torch.nn", "probly.layers.torch"]
    for k in victims:
        sys.modules.pop(k, None)
    yield
    for k in victims:
        sys.modules.pop(k, None)

def install_fakes():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn.Linear = FakeLinear
    torch.nn = nn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn

    layers_torch = types.ModuleType("probly.layers.torch")
    layers_torch.NormalInverseGammaLinear = FakeNormalInverseGammaLinear
    sys.modules["probly.layers.torch"] = layers_torch

    common = types.ModuleType("probly.transformation.evidential.regression.common")
    common.REPLACED_LAST_LINEAR = FakeGlobalVariable[bool]("X", "", default=False)
    common.register_calls = []
    def register(cls, traverser):
        common.register_calls.append((cls, traverser))
    common.register = register
    sys.modules["probly.transformation.evidential.regression.common"] = common

    pkg = types.ModuleType("probly.transformation.evidential.regression")
    sys.modules["probly.transformation.evidential.regression"] = pkg

    return common

def import_target():
    return importlib.import_module("probly.transformation.evidential.regression.torch")

def test_register_called_on_import():
    common = install_fakes()
    mod = import_target()
    assert common.register_calls
    cls, traverser = common.register_calls[-1]
    assert cls is sys.modules["torch.nn"].Linear
    assert traverser is mod.replace_last_torch_nig

def test_replace_last_torch_nig_sets_state_and_builds_layer():
    install_fakes()
    mod = import_target()
    state = {}
    lin = sys.modules["torch.nn"].Linear(4, 1, bias=True, device="cuda")
    layer, new_state = mod.replace_last_torch_nig(lin, state)
    assert state[sys.modules["probly.transformation.evidential.regression.common"].REPLACED_LAST_LINEAR] is True
    assert isinstance(layer, FakeNormalInverseGammaLinear)
    assert layer.in_features == 4
    assert layer.out_features == 1
    assert layer.device == "cuda"
    assert layer.bias is True

