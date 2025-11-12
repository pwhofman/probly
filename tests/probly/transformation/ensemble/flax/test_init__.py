import importlib

def test_import_and_functions_exist():
    # Modul importieren
    module = importlib.import_module("probly.transformation.ensemble.flax")

    # Prüfen, ob wichtige Funktionen existieren
    assert hasattr(module, "ensemble")
    assert hasattr(module, "register")

    # Prüfen, ob sie aufrufbar sind (also Funktionen oder Objekte mit __call__)
    assert callable(module.ensemble)
    assert callable(module.register)
