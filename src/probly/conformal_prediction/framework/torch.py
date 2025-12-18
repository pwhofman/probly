"""TorchWrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from probly.conformal_prediction.methods.common import PredictiveModel


class TorchModelWrapper(PredictiveModel):
    """Wrapper to make PyTorch models compatible with the PredictiveModel protocol."""

    def __init__(self, torch_model: nn.Module, device: str | None = None) -> None:
        """Initialize the wrapper with a trained PyTorch model."""
        self.torch_model = torch_model

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.torch_model.to(self.device)
        self.torch_model.eval()

    def _to_numpy(self, data: object) -> npt.NDArray[np.generic]:
        """Convert a tensor or array like object to a NumPy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Compute model predictions as probabilities (NumPy array)."""
        # ensure tensor input
        x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

        x_tensor = x_tensor.to(self.device)
        self.torch_model.eval()

        with torch.no_grad():
            outputs = self.torch_model(x_tensor)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # convert to probabilities
            if outputs.ndim == 1 or outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs)
                if probs.ndim == 1:
                    probs = probs.unsqueeze(1)
            else:
                probs = torch.softmax(outputs, dim=1)

        return np.asarray(probs.cpu().numpy(), dtype=np.float32)
