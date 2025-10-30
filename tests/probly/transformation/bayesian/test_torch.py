from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from torch import nn

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


def test_if_replace_torch_bayesian_linear_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure replace_torch_bayesian_linear correctly wraps nn.Linear."""
    called: dict[str, Any] = {}

    class FakeBayesLinear:
        def __init__(
            self,
            obj: nn.Linear,
            use_base_weights: bool,
            posterior_std: float,
            prior_mean: float,
            prior_std: float,
        ) -> None:
            called.update(locals())

    # Replace BayesLinear with fake version
    monkeypatch.setattr(t, "BayesLinear", FakeBayesLinear)

    base = nn.Linear(5, 3)
    result = t.replace_torch_bayesian_linear(base, True, 0.3, 0.1, 0.2)

    assert isinstance(result, FakeBayesLinear)
    assert called["obj"] is base
    assert called["use_base_weights"] is True
    assert called["posterior_std"] == 0.3
    assert called["prior_mean"] == 0.1
    assert called["prior_std"] == 0.2


def test_if_replace_torch_bayesian_conv2d_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure replace_torch_bayesian_conv2d correctly wraps nn.Conv2d."""
    called: dict[str, Any] = {}

    class FakeBayesConv2d:
        def __init__(
            self,
            obj: nn.Conv2d,
            use_base_weights: bool,
            posterior_std: float,
            prior_mean: float,
            prior_std: float,
        ) -> None:
            called.update(locals())

    # Replace BayesConv2d with fake version
    monkeypatch.setattr(t, "BayesConv2d", FakeBayesConv2d)

    base = nn.Conv2d(5, 3, 3)
    result = t.replace_torch_bayesian_conv2d(base, True, 0.3, 0.1, 0.2)

    assert isinstance(result, FakeBayesConv2d)
    assert called["obj"] is base
    assert called["use_base_weights"] is True
    assert called["posterior_std"] == 0.3
    assert called["prior_mean"] == 0.1
    assert called["prior_std"] == 0.2
