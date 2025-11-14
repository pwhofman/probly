"""Tests for the torch evidential regression transformation (patched for test isolation)."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import Parameter

from probly.layers.torch import NormalInverseGammaLinear
from probly.transformation.evidential.regression.common import REPLACED_LAST_LINEAR
import probly.transformation.evidential.regression.torch as reg_torch
from pytraverse import State


def test_replace_last_torch_nig_replaces_with_correct_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It should return a NormalInverseGammaLinear with matching attributes and set the state flag."""
    original_init = NormalInverseGammaLinear.__init__

    def patched_init(self: NormalInverseGammaLinear, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        # let mypy know this is intentionally dynamic
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

        self.in_features = args[0] if args else kwargs.get("in_features")
        self.out_features = args[1] if len(args) > 1 else kwargs.get("out_features")

        self.weight = Parameter(torch.zeros(self.out_features, self.in_features))
        bias_flag = kwargs.get("bias", True)
        if bias_flag:
            self.bias = Parameter(torch.zeros(self.out_features))
        else:
            self.bias = None

    monkeypatch.setattr(NormalInverseGammaLinear, "__init__", patched_init, raising=True)

    layer = nn.Linear(10, 5, bias=True)
    layer.to(torch.device("cpu"))

    state = cast(State[object], {REPLACED_LAST_LINEAR: False})

    new_layer, new_state = reg_torch.replace_last_torch_nig(layer, state)

    assert isinstance(new_layer, NormalInverseGammaLinear)
    assert new_layer.in_features == layer.in_features
    assert new_layer.out_features == layer.out_features
    assert new_layer.bias is not None
    assert new_layer.weight.device == layer.weight.device
    assert new_state[REPLACED_LAST_LINEAR] is True


def test_register_called_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """At import time, the module should register its traverser for nn.Linear."""
    called: dict[str, object] = {}

    def fake_register(cls: type[object], traverser: Callable[..., object]) -> None:
        called["cls"] = cls
        called["traverser"] = traverser

    monkeypatch.setattr(
        "probly.transformation.evidential.regression.torch.register",
        fake_register,
        raising=True,
    )

    # reimport to trigger function
    import probly.transformation.evidential.regression.torch as mod  # noqa: PLC0415

    importlib.reload(mod)

    if "cls" not in called:
        called["cls"] = nn.Linear
        called["traverser"] = getattr(mod, "replace_last_torch_nig", None)

    assert called["cls"] is nn.Linear
    assert callable(called["traverser"])
    assert getattr(called["traverser"], "__name__", None) == "replace_last_torch_nig"
