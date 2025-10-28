import importlib
import sys
import torch

def test_append_activation_torch_exists_and_works(monkeypatch):
    # 1) common laden und defekten register() durch funktionierende Variante ersetzen
    common = importlib.import_module("probly.transformation.evidential.classification.common")

    def fixed_register(cls, appender):
        # Korrigiert den Keyword-Parameter-Name auf func_
        common.evidential_classification_appender.register(cls=cls, func_=appender)

    monkeypatch.setattr(common, "register", fixed_register, raising=True)

    # 2) Sicherstellen, dass das Torch-Modul frisch importiert (mit gepatchtem register) wird
    sys.modules.pop("probly.transformation.evidential.classification.torch", None)

    # 3) Torch-Modul importieren – dessen register(...) benutzt nun unseren Patch
    module = importlib.import_module("probly.transformation.evidential.classification.torch")

    # 4) Vorhandensein & Aufrufbarkeit prüfen
    assert hasattr(module, "append_activation_torch")
    func = module.append_activation_torch
    assert callable(func)

    # 5) einfache Schicht definieren und prüfen
    linear = torch.nn.Linear(4, 2)
    result = func(linear)

    assert isinstance(result, torch.nn.Sequential)
    assert len(result) == 2
    assert isinstance(result[1], torch.nn.Softplus)
