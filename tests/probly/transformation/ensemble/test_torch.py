from __future__ import annotations

import torch
from torch import nn

# Wir testen direkt die Implementierung in torch.py
from probly.transformation.ensemble.torch import generate_torch_ensemble


def _w_b(layer: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    """Hilfsfunktion: Weight & Bias (detach + clone) für stabile Vergleiche."""
    return layer.weight.detach().clone(), layer.bias.detach().clone()


def test_torch_ensemble_without_reset_passes() -> None:
    """
    reset_params=False:
    - gibt eine nn.ModuleList zurück
    - Anzahl der Mitglieder stimmt
    - Mitglieder sind echte Kopien (andere Objekte)
    - Parameterwerte sind identisch zum Basismodell
    """
    base = nn.Linear(4, 2)


    members = generate_torch_ensemble(base, num_members=3, reset_params=False)

    assert isinstance(members, nn.ModuleList)
    assert len(members) == 3
    for m in members:
        assert m is not base

    w0, b0 = _w_b(base)
    w1, b1 = _w_b(members[0])


    assert torch.allclose(w0, w1)
    assert torch.allclose(b0, b1)


def test_torch_ensemble_with_reset_passes() -> None:
    """
    reset_params=True:
    - gibt eine nn.ModuleList zurück
    - Anzahl der Mitglieder stimmt
    - Parameter der Kopien sind (typischerweise) ungleich zum Basismodell,
      da reset_parameters() neu initialisiert.
    """
    base = nn.Linear(4, 2)
    w0, b0 = _w_b(base)

    members = generate_torch_ensemble(base, num_members=3, reset_params=True)

    assert isinstance(members, nn.ModuleList)
    assert len(members) == 3

    w1, b1 = _w_b(members[0])

    # nach Reset: Werte unterscheiden sich normalerweise
    assert not (torch.allclose(w0, w1) and torch.allclose(b0, b1))
