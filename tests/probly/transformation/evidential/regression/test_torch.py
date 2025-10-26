import sys
import types
import importlib
from pathlib import Path
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
    victims = [
        "torch", "torch.nn",
        "probly.layers", "probly.layers.torch",
        "probly.transformation.evidential.regression.common",
        "probly.transformation.evidential.regression.torch",
    ]
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

    layers_pkg = types.ModuleType("probly.layers")
    layers_pkg.__path__ = []
    sys.modules["probly.layers"] = layers_pkg
    layers_torch = types.ModuleType("probly.layers.torch")
    layers_torch.NormalInverseGammaLinear = FakeNormalInverseGammaLinear
    sys.modules["probly.layers.torch"] = layers_torch

    common = types.ModuleType("probly.transformation.evidential.regression.common")
    class _GV(FakeGlobalVariable): pass
    common.REPLACED_LAST_LINEAR = _GV[bool]("REPLACED_LAST_LINEAR", "", default=False)
    common.register_calls = []
    def register(cls, traverser):
        common.register_calls.append((cls, traverser))
    common.register = register
    sys.modules["probly.transformation.evidential.regression.common"] = common

    probly_pkg = types.ModuleType("probly")
    probly_pkg.__path__ = [str(Path.cwd() / "src" / "probly")]
    sys.modules["probly"] = probly_pkg

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

def test_replace_last_torch_nig_sets_state_and_builds_layer_bias_true():
    install_fakes()
    mod = import_target()
    state = {}
    lin = sys.modules["torch.nn"].Linear(4, 1, bias=True, device="cuda")
    layer, new_state = mod.replace_last_torch_nig(lin, state)
    assert new_state is state
    assert state[sys.modules["probly.transformation.evidential.regression.common"].REPLACED_LAST_LINEAR] is True
    assert isinstance(layer, FakeNormalInverseGammaLinear)
    assert layer.in_features == 4
    assert layer.out_features == 1
    assert layer.device == "cuda"
    assert layer.bias is True

def test_replace_last_torch_nig_builds_layer_bias_false():
    install_fakes()
    mod = import_target()
    state = {}
    lin = sys.modules["torch.nn"].Linear(5, 2, bias=False, device="cpu")
    layer, _ = mod.replace_last_torch_nig(lin, state)
    assert isinstance(layer, FakeNormalInverseGammaLinear)
    assert layer.in_features == 5
    assert layer.out_features == 2
    assert layer.device == "cpu"
    assert layer.bias is False
