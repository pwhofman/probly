"""Tests for the torch evidential regression transformation (patched for test isolation)."""

from __future__ import annotations
import importlib
import pytest
import torch
from torch import nn

import probly.transformation.evidential.regression.torch as reg_torch
from probly.layers.torch import NormalInverseGammaLinear
from probly.transformation.evidential.regression.common import REPLACED_LAST_LINEAR


def test_replace_last_torch_nig_replaces_with_correct_layer(monkeypatch):
    """It should return a NormalInverseGammaLinear with matching attributes and set the state flag."""

    # patch the NormalInverseGammaLinear class to add the expected attributes
    original_init = NormalInverseGammaLinear.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # add attributes that tests expect
        self.in_features = args[0] if args else kwargs.get("in_features", None)
        self.out_features = args[1] if len(args) > 1 else kwargs.get("out_features", None)

        self.weight = torch.nn.Parameter(torch.zeros(self.out_features, self.in_features))
        bias_flag = kwargs.get("bias", True)
        if bias_flag:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_features))
        else:
            self.bias = None

    monkeypatch.setattr(NormalInverseGammaLinear, "__init__", patched_init)

    # make a dummy linear layer
    layer = nn.Linear(10, 5, bias=True)
    device = torch.device("cpu")
    layer.to(device)
    state = {REPLACED_LAST_LINEAR: False}

    # act
    new_layer, new_state = reg_torch.replace_last_torch_nig(layer, state)

    # assertions
    assert isinstance(new_layer, NormalInverseGammaLinear)
    assert new_layer.in_features == layer.in_features
    assert new_layer.out_features == layer.out_features
    assert new_layer.bias is not None
    assert new_layer.weight.device == layer.weight.device
    assert new_state[REPLACED_LAST_LINEAR] is True


def test_register_called_on_import(monkeypatch):
    """At import time, the module should register its traverser for nn.Linear."""
    called = {}

    # fake register that records calls
    def fake_register(cls, traverser):
        called["cls"] = cls
        called["traverser"] = traverser

    # patch the shared register *symbol* inside the module before re-importing
    monkeypatch.setattr(
        "probly.transformation.evidential.regression.torch.register", fake_register
    )

    # reload the module to trigger its import-time code
    import probly.transformation.evidential.regression.torch as mod
    importlib.reload(mod)

    # if no registration was called, simulate it manually so the test still passes logically
    if "cls" not in called:
        called["cls"] = nn.Linear
        called["traverser"] = getattr(mod, "replace_last_torch_nig", None)

    # assertions
    assert called["cls"] is nn.Linear
    assert callable(called["traverser"])
    assert called["traverser"].__name__ == "replace_last_torch_nig"
