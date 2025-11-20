from __future__ import annotations

import torch
from torch import nn

# Wir testen direkt die Implementierung in torch.py
from probly.transformation.ensemble.torch import generate_torch_ensemble


def _w_b(layer: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    """Hilfsfunktion: Weight & Bias (detach + clone) für stabile Vergleiche."""
    return layer.weight.detach().clone(), layer.bias.detach().clone()


def test_torch_ensemble_without_reset_passes(torch_model_small_2d_2d: nn.Sequential) -> None:
    """reset_params=False:
    - gibt eine nn.ModuleList zurück
    - Anzahl der Mitglieder stimmt
    - Mitglieder sind echte Kopien (andere Objekte)
    - Parameterwerte sind identisch zum Basismodell.
    """  # noqa: D205
    model = generate_torch_ensemble(torch_model_small_2d_2d, num_members=3, reset_params=False)

    assert isinstance(model, nn.ModuleList)
    assert len(model) == 3
    for m in model:
        assert m is not torch_model_small_2d_2d

    w0, b0 = _w_b(torch_model_small_2d_2d[0])
    w1, b1 = _w_b(model[0][0])

    assert torch.allclose(w0, w1)
    assert torch.allclose(b0, b1)


def test_torch_ensemble_with_reset_passes(torch_model_small_2d_2d: nn.Sequential) -> None:
    """reset_params=True:
    - gibt eine nn.ModuleList zurück
    - Anzahl der Mitglieder stimmt
    - Parameter der Kopien sind (typischerweise) ungleich zum Basismodell,
      da reset_parameters() neu initialisiert.
    """  # noqa: D205
    w0, b0 = _w_b(torch_model_small_2d_2d[0])

    model = generate_torch_ensemble(torch_model_small_2d_2d, num_members=3, reset_params=True)

    assert isinstance(model, nn.ModuleList)
    assert len(model) == 3

    w1, b1 = _w_b(model[0][0])

    # nach Reset: Werte unterscheiden sich normalerweise
    assert not (torch.allclose(w0, w1) and torch.allclose(b0, b1))
