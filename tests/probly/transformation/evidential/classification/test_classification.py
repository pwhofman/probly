import pytest
from torch import nn

from probly.transformation.evidential.classification import common
from probly.transformation.evidential.classification import torch as torch_module
from probly.transformation.evidential.classification.common import (
    evidential_classification,
    register
)


def test_append_activation_torch_returns_sequential():
    """Testet, ob append_activation_torch ein nn.Sequential zurückgibt mit Softplus."""
    module = nn.Linear(10, 5)
    seq = torch_module.append_activation_torch(module)
    
    # Prüfe Typ
    assert isinstance(seq, nn.Sequential)
    # Prüfe, dass letzte Schicht Softplus ist
    assert isinstance(seq[-1], nn.Softplus)


def test_register_and_evidential_classification():
    """Testet die Registration und evidential_classification-Funktion."""
    
    class DummyModule(nn.Module):
        pass

    # Dummy-Appender definieren
    def dummy_appender(obj: nn.Module) -> nn.Module:
        return nn.Sequential(obj, nn.ReLU())

    # DummyModule registrieren
    register(DummyModule, dummy_appender)

    # DummyModule erstellen
    dummy = DummyModule()
    
    # Prüfe, dass evidential_classification den Appender benutzt
    result = evidential_classification(dummy)
    # Letzte Schicht sollte ReLU sein
    assert isinstance(result[-1], nn.ReLU)
