from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from torch import nn

from probly.layers.torch import BayesConv2d, BayesLinear
from probly.transformation.bayesian import common, torch as t


def test_if_register_is_called_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check if register() is called automatically when torch module is imported."""
    called: list[tuple[type[Any], Any]] = []

    def fake_register(cls: type[Any], trv: object) -> None:
        called.append((cls, trv))

    # Patch common.register with our fake function
    monkeypatch.setattr(common, "register", fake_register, raising=True)

    # Reimport module to trigger register() calls
    modname = "probly.transformation.bayesian.torch"
    sys.modules.pop(modname, None)
    importlib.import_module(modname)

    # Convert list of tuples to dict for easier lookup
    called2 = dict(called)

    # Both nn.Linear and nn.Conv2d should have been registered
    assert nn.Linear in called2
    assert nn.Conv2d in called2


def test_if_replace_torch_bayesian_linear_works(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Ensure replace_torch_bayesian_linear correctly wraps nn.Linear."""
    # switch the layers from linear to bayesian within our model and save the bayesian ones in bayesmodel_arr
    bayesmodel_arr = nn.Sequential()
    for i in range(len(torch_model_small_2d_2d)):
        if isinstance(torch_model_small_2d_2d[i], nn.Linear):
            bayesmodel_arr.append(
                t.replace_torch_bayesian_linear(
                    torch_model_small_2d_2d[i],
                    True,
                    0.5,
                    1.3,
                    0.4,
                ),
            )

    for module in range(len(torch_model_small_2d_2d)):
        if isinstance(module, nn.Linear):
            assert isinstance(bayesmodel_arr[i], BayesLinear)


def test_if_replace_torch_bayesian_conv2d_works(torch_conv_linear_model: nn.Sequential) -> None:
    """Ensure replace_torch_bayesian_conv2d correctly wraps nn.Conv2d."""
    # switch the layers from conv2d to bayesian within our model and save the bayesian ones in bayesmodel_arr
    bayesmodel_arr = nn.Sequential()
    for i in range(len(torch_conv_linear_model)):
        if isinstance(torch_conv_linear_model[i], nn.Conv2d):
            bayesmodel_arr.append(
                t.replace_torch_bayesian_conv2d(
                    torch_conv_linear_model[i],
                    True,
                    0.5,
                    1.3,
                    0.4,
                ),
            )

    for module in range(len(torch_conv_linear_model)):
        if isinstance(module, nn.Conv2d):
            assert isinstance(bayesmodel_arr[i], BayesConv2d)
