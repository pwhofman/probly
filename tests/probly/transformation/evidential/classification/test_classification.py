from __future__ import annotations

from torch import nn

from probly.transformation.evidential.classification import torch as torch_module
from probly.transformation.evidential.classification.common import (
    evidential_classification,
    register,
)


def test_append_activation_torch_returns_sequential() -> None:
    """Testet, ob append_activation_torch ein nn.Sequential zurückgibt mit Softplus."""
    module = nn.Linear(10, 5)
    seq = torch_module.append_activation_torch(module)

    assert isinstance(seq, nn.Sequential)
    assert isinstance(seq[-1], nn.Softplus)


def test_register_and_evidential_classification() -> None:
    """Testet die Registration und evidential_classification-Funktion."""

    class DummyModule(nn.Module):
        pass

    def dummy_appender(obj: nn.Module) -> nn.Module:
        return nn.Sequential(obj, nn.ReLU())

    register(DummyModule, dummy_appender)

    dummy = DummyModule()

    result = evidential_classification(dummy)

    assert isinstance(result[-1], nn.ReLU)
