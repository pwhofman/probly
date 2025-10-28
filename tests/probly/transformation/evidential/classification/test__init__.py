# tests/probly/transformation/evidential/classification/test_init.py

import importlib

def test_import_and_functions_exist():
    # Modul laden
    module = importlib.import_module("probly.transformation.evidential.classification")

    # prüfen, ob es die Funktionen gibt
    assert hasattr(module, "evidential_classification")
    assert hasattr(module, "register")

    # prüfen, ob sie aufrufbar sind (also Funktionen)
    assert callable(module.evidential_classification)
    assert callable(module.register)
