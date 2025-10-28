import importlib
import types
import importlib as _il
import pytest
import types as _types
from torch._dynamo.polyfills import sys

# Wir testen das vorgegebene Modul
common = importlib.import_module(
    "probly.transformation.evidential.classification.common"
)

def test_api_surface_exists_and_callable():
    # Funktionen existieren
    assert hasattr(common, "evidential_classification_appender")
    assert hasattr(common, "register")
    assert hasattr(common, "evidential_classification")

    # und sind aufrufbar
    assert callable(common.evidential_classification_appender)
    assert callable(common.register)
    assert callable(common.evidential_classification)



