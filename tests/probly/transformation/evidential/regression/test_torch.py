import importlib
import sys
import pytest
import torch
from torch import nn


def _fresh_import():
    for k in [
        "probly.transformation.evidential.regression.torch",
    ]:
        sys.modules.pop(k, None)
    import probly.transformation.evidential.regression.common as common
    import probly.transformation.evidential.regression.torch as mod
    importlib.reload(common)
    importlib.reload(mod)
    return common, mod


def test_register_called_on_import():
    import probly.transformation.evidential.regression.common as common
    captured = {}
    def _capture(*args, **kwargs):
        if kwargs:
            captured["cls"] = kwargs.get("cls")
            captured["traverser"] = kwargs.get("traverser")
        else:
            captured["cls"] = args[0]
            captured["traverser"] = args[1]
    original = common.register
    try:
        common.register = _capture  # type: ignore[attr-defined]
        _, mod = _fresh_import()
    finally:
        common.register = original  # type: ignore[attr-defined]
    assert captured["cls"] is nn.Linear
    assert captured["traverser"].__name__ == "replace_last_torch_nig"


def test_replace_last_torch_nig_sets_flag_and_builds_layer_cpu():
    common, mod = _fresh_import()
    import probly.layers.torch as layers
    state = {}
    lin = nn.Linear(4, 1, bias=True).to("cpu")
    layer, new_state = mod.replace_last_torch_nig(lin, state)
    assert new_state is state
    assert state[common.REPLACED_LAST_LINEAR] is True
    assert isinstance(layer, layers.NormalInverseGammaLinear)
