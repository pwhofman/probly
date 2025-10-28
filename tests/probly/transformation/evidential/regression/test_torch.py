import importlib
import pytest
import torch
from torch import nn


def test_register_called_on_import(mocker):
    import probly.transformation.evidential.regression.common as common
    import probly.transformation.evidential.regression.torch as mod
    importlib.reload(common)
    spy = mocker.spy(common, "register")
    importlib.reload(mod)
    assert spy.call_count >= 1
    args, kwargs = spy.call_args
    assert kwargs["cls"] is nn.Linear
    assert kwargs["traverser"] is mod.replace_last_torch_nig


@pytest.mark.parametrize("bias", [True, False])
def test_replace_last_torch_nig_sets_flag_and_builds_layer_cpu(bias):
    import probly.transformation.evidential.regression.common as common
    import probly.transformation.evidential.regression.torch as mod
    import probly.layers.torch as layers
    importlib.reload(common)
    importlib.reload(mod)
    state = {}
    lin = nn.Linear(4, 1, bias=bias).to("cpu")
    layer, new_state = mod.replace_last_torch_nig(lin, state)
    assert new_state is state
    assert state[common.REPLACED_LAST_LINEAR] is True
    assert isinstance(layer, layers.NormalInverseGammaLinear)
    assert getattr(layer, "in_features", None) == 4
    assert getattr(layer, "out_features", None) == 1
    if hasattr(layer, "parameters"):
        for p in layer.parameters():
            assert p.device.type == "cpu"
    if hasattr(layer, "bias"):
        assert (layer.bias is None) is (bias is False)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_replace_last_torch_nig_sets_flag_and_builds_layer_cuda(bias):
    import probly.transformation.evidential.regression.common as common
    import probly.transformation.evidential.regression.torch as mod
    import probly.layers.torch as layers
    importlib.reload(common)
    importlib.reload(mod)
    state = {}
    lin = nn.Linear(5, 2, bias=bias).to("cuda")
    layer, new_state = mod.replace_last_torch_nig(lin, state)
    assert new_state is state
    assert state[common.REPLACED_LAST_LINEAR] is True
    assert isinstance(layer, layers.NormalInverseGammaLinear)
    assert getattr(layer, "in_features", None) == 5
    assert getattr(layer, "out_features", None) == 2
    if hasattr(layer, "parameters"):
        for p in layer.parameters():
            assert p.device.type == "cuda"
    if hasattr(layer, "bias"):
        assert (layer.bias is None) is (bias is False)


def test_only_last_linear_is_replaced_behaviorally_cpu():
    import probly.transformation.evidential.regression.common as common
    import probly.layers.torch as layers
    importlib.reload(common)
    model = nn.Sequential(
        nn.Linear(4, 3, bias=True),
        nn.ReLU(),
        nn.Linear(3, 1, bias=False),
    ).to("cpu")
    transformed = common.evidential_regression(model)
    assert isinstance(transformed[0], nn.Linear)
    assert isinstance(transformed[1], nn.ReLU)
    assert isinstance(transformed[2], layers.NormalInverseGammaLinear)
    assert getattr(transformed[2], "in_features", None) == 3
    assert getattr(transformed[2], "out_features", None) == 1
    if hasattr(transformed[2], "bias"):
        assert transformed[2].bias is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_only_last_linear_is_replaced_behaviorally_cuda():
    import probly.transformation.evidential.regression.common as common
    import probly.layers.torch as layers
    importlib.reload(common)
    model = nn.Sequential(
        nn.Linear(6, 4, bias=True).to("cuda"),
        nn.ReLU(),
        nn.Linear(4, 2, bias=True).to("cuda"),
    )
    transformed = common.evidential_regression(model)
    assert isinstance(transformed[0], nn.Linear)
    assert isinstance(transformed[1], nn.ReLU)
    assert isinstance(transformed[2], layers.NormalInverseGammaLinear)
    assert getattr(transformed[2], "in_features", None) == 4
    assert getattr(transformed[2], "out_features", None) == 2
    if hasattr(transformed[2], "parameters"):
        for p in transformed[2].parameters():
            assert p.device.type == "cuda"
