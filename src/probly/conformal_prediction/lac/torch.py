"""PyTorch implementation of Local Aggregative Conformal (LAC) prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

import probly.conformal_prediction.lac.common as common_lac  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


class TorchModelWrapper:
    """Wrapper to make PyTorch models compatible with PredictiveModel protocol."""

    def __init__(self, torch_model: nn.Module) -> None:
        """Initialize with a PyTorch model."""
        self.torch_model = torch_model
        self.torch_model.eval()

    def predict(self, x: Sequence[Any]) -> np.ndarray:
        """Predict method for PredictiveModel protocol."""
        # convert input to torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            # call the model
            output = self.torch_model(x)

            # convert output to numpy
            output_np = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)

            # check if output needs softmax
            if output_np.shape[1] > 1 and (np.any(output_np > 1.0) or np.any(output_np < 0.0)):
                # use softmax to convert logits to probabilities
                exp_output = np.exp(output_np - np.max(output_np, axis=1, keepdims=True))
                output_np = exp_output / exp_output.sum(axis=1, keepdims=True)

            return output_np


class LAC(common_lac.LAC):  # type: ignore[name-defined]
    """PyTorch-specific implementation of LAC.

    Handles automated conversion between PyTorch Tensors and Numpy arrays,
    ensuring seamless integration with PyTorch models while preserving device placement.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize with a PyTorch model."""
        wrapped_model = TorchModelWrapper(model)
        super().__init__(wrapped_model)
        self.torch_model = model

    def _to_numpy(self, data: object) -> npt.NDArray[np.generic]:
        """Helper: Converts Tensor (CPU/GPU) to Numpy."""
        if isinstance(data, torch.Tensor):
            # Explicitly allow Any return type for numpy conversion
            return data.detach().cpu().numpy()  # type: ignore[no-any-return]
        return np.asarray(data)

    def calibrate(  # type: ignore[override]
        self,
        x_cal: object,
        y_cal: object,
        significance_level: float,
    ) -> float | None:
        """Calibrate using PyTorch tensors.

        Converts inputs to numpy and delegates to the base implementation.
        """
        x_np = self._to_numpy(x_cal)
        y_np = self._to_numpy(y_cal)

        # Delegate to base implementation
        super().calibrate(x_np, y_np, significance_level)

        # Explicitly return the threshold for testing purposes
        return self.threshold

    def predict(
        self,
        x: Sequence[Any],
        significance_level: float,
    ) -> torch.Tensor:
        """Generate prediction sets for PyTorch inputs.

        Returns:
            torch.Tensor: Boolean tensor of shape (n_samples, n_classes)
                          on the same device as input x.
        """
        # Maintain API consistency with base class signature
        _ = significance_level

        if not self.is_calibrated or self.threshold is None:
            msg = "Predictor is not calibrated. Call calibrate() first."
            raise RuntimeError(msg)

        # 1. Get model predictions
        probas = self.model.predict(x)

        # Store device for final output
        target_device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")

        # 2. Convert to Numpy for core logic
        probas_np = self._to_numpy(probas)

        # 3. LAC Logic (Numpy based)
        # Threshold logic: score <= threshold <=> 1 - p <= t <=> p >= 1 - t
        prob_threshold = 1.0 - self.threshold

        # Create initial prediction sets
        prediction_sets_np = probas_np >= prob_threshold

        # Apply Accretive Completion to fix empty sets (Null Regions)
        final_sets_np = common_lac.accretive_completion(prediction_sets_np, probas_np)  # type: ignore[attr-defined]

        # 4. Convert back to PyTorch and move to original device
        return torch.from_numpy(final_sets_np).to(target_device)
