"""torch wrapper."""

# probly/conformal_prediction/framework/torch.py
# oder: probly/conformal_prediction/backends/torch.py
# -------------------------------------------------------------------
# AUS: lac/torch.py::TorchModelWrapper [file:12]
# AUS: aps/torch.py::__init__, _get_probs (Device-Handling, Softmax) [file:15]
# NEU: vereinheitlichter TorchModelWrapper für alle Scores

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


class TorchModelWrapper:
    """Wrapper, um PyTorch-Modelle für die Score-Klassen nutzbar zu machen.

    Verantwortlichkeiten:
    - Modell auf das richtige Device verschieben (CPU / CUDA).
    - eval()-Modus setzen.
    - Eingaben (Listen/np.ndarray/Tensors) in torch.Tensor konvertieren.
    - Vorwärtslauf ausführen.
    - Ausgaben in Wahrscheinlichkeiten umwandeln (Softmax/Sigmoid).
    - Ergebnis als numpy-Array (n_samples, n_classes) zurückgeben.

    Dieser Wrapper ersetzt die Kombination aus:
    - APSPredictor.__init__ + _get_probs (aps/torch.py) [file:15]
    - TorchModelWrapper (lac/torch.py) [file:12]
    """

    def __init__(self, torch_model: nn.Module, device: str | None = None) -> None:
        """Initialisiere Wrapper mit einem trainierten PyTorch-Modell.

        Args:
            torch_model: PyTorch nn.Module.
            device: Optionaler Device-String, z.B. "cuda" oder "cpu".
                    Wenn None: "cuda", falls verfügbar, sonst "cpu".
        """
        self.torch_model = torch_model

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.torch_model.to(self.device)
        self.torch_model.eval()

    def _to_numpy(self, data: object) -> npt.NDArray[np.generic]:
        """Hilfsfunktion: Torch-Tensor oder Array nach numpy konvertieren.

        AUS: lac/torch.py::_to_numpy [file:12]
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()  # type: ignore[no-any-return]
        return np.asarray(data)

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Berechne Modellvorhersage als Wahrscheinlichkeiten (numpy).

        Entspricht der Logik aus APSPredictor._get_probs (aps/torch.py),
        verallgemeinert auf alle Scores. [file:15]

        Args:
            x: Eingabedaten (Sequence, numpy-Array oder Torch-Tensor).

        Returns:
            numpy-Array mit Wahrscheinlichkeiten, shape (n_samples, n_classes).
        """
        # 1. Eingabe in Tensor wandeln, falls nötig
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)
        self.torch_model.eval()

        with torch.no_grad():
            outputs = self.torch_model(x)

            # Tuple-Ausgaben (z.B. (logits, extra)) behandeln
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # 2. In Wahrscheinlichkeiten umwandeln
            if outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)
            else:
                # Binary-Case: Sigmoid
                probs = torch.sigmoid(outputs)

        # 3. Nach numpy konvertieren
        return np.asarray(probs.cpu().numpy(), dtype=np.float32)
