# tests/probly/transformation/evidential/classification/test_torch.py
import importlib
import torch

def test_append_activation_torch_exists_and_works():
    # torch-Implementierung importieren (registriert beim Import)
    mod = importlib.import_module(
        "probly.transformation.evidential.classification.torch"
    )

    # Existenz der Funktion prüfen (einsteigerfreundlich)
    assert hasattr(mod, "append_activation_torch")
    assert callable(mod.append_activation_torch)

    # Jetzt die öffentliche API nutzen -> beweist, dass die Registrierung funktioniert
    from probly.transformation.evidential.classification import common

    base = torch.nn.Linear(4, 2)
    wrapped = common.evidential_classification(base)

    assert isinstance(wrapped, torch.nn.Sequential)
    assert len(wrapped) == 2
    assert isinstance(wrapped[1], torch.nn.Softplus)
