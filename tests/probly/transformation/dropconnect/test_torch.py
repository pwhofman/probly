from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from torch import nn

from probly.transformation.dropconnect import common, torch as t


def test_if_register_is_called_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check if register() is called automatically when torch module is imported."""
    called: list[tuple[type[Any], Any]] = []

    def fake_register(cls: type[Any], trv: object) -> None:
        called.append((cls, trv))

    # Patch common.register with our fake function
    monkeypatch.setattr(common, "register", fake_register, raising=True)

    # Reimport module to trigger register() calls
    modname = "probly.transformation.dropconnect.torch"
    sys.modules.pop(modname, None)
    importlib.import_module(modname)

    # Convert list of tuples to dict for easier lookup
    called2 = dict(called)

    # nn.Linear should have been registered
    assert nn.Linear in called2


def test_if_replace_torch_dropconnect_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure replace_torch_dropconnect correctly wraps nn.Linear."""
    called: dict[str, Any] = {}

    class FakeDropConnectLinear:
        def __init__(
            self,
            obj: nn.Linear,
            p: float,
        ) -> None:
            called.update(locals())

    # Replace BayesLinear with fake version
    monkeypatch.setattr(t, "DropConnectLinear", FakeDropConnectLinear)

    base = nn.Linear(5, 3)
    result = t.replace_torch_dropconnect(base, 0.3)

    assert isinstance(result, FakeDropConnectLinear)
    assert called["obj"] is base
    assert called["p"] == 0.3
