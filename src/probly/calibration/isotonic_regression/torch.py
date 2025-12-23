"""Torch Implementations for Isotonic Regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.isotonic import IsotonicRegression
from torch import Tensor, cat, device as TorchDevice, nn, no_grad, sigmoid, softmax, tensor, unique

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class IsotonicRegressionCalibrator:
    """Class for the isotonic regression calibration."""

    def __init__(self, base_model: nn.Module, device: TorchDevice) -> None:
        """Set up the wrapper.

        Args:
            base_model: The base model to calibrate
            device: The device that torch uses for the model

        """
        self.model = base_model
        self.device = device
        self.calibrator: list[IsotonicRegression] = []

    def fit(self, calibration_set: DataLoader) -> None:
        """Fit the regression function to the model outputs.

        Args:
            calibration_set: The set that should be used for the calibration

        """
        # get logits for calibration data
        logits_and_labels = self._extract_logits_and_labels(calibration_set)
        logits_cal = logits_and_labels[0]
        labels_cal = logits_and_labels[1]

        # Identify number of classes and learn regression function
        self.classes_ = unique(labels_cal).cpu().numpy()
        num_classes = len(self.classes_)

        if num_classes <= 2:
            pos_label = 1
            labels_bin = (labels_cal == pos_label).float()

            scores = self._get_binary_scores(logits_cal)
            probabilities = sigmoid(scores).cpu()

            iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso_reg.fit(probabilities.numpy(), labels_bin.numpy())
            self.calibrator = [iso_reg]

        else:
            probabilities = softmax(logits_cal, dim=1).cpu().numpy()

            for c in self.classes_:
                iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso_reg.fit(probabilities[:, c], (labels_cal == c).numpy())
                self.calibrator.append(iso_reg)

    def predict(self, x: Tensor) -> Tensor:
        """Make calibrated predictions on the input x.

        Args:
            x: The input for the model to make predictions on

        Returns:
            calibrated_probs: The calibrated probabilities for the prediction

        """
        self.model.eval()

        with no_grad():
            x = x.to(self.device)
            logits = self.model(x)

            if self.calibrator is None:
                calibrator_none_message = "Calibrator has not been fitted yet. Call fit method first."
                raise RuntimeError(calibrator_none_message)

            # Case Multiclass model
            if len(self.calibrator) >= 2:
                probabilities = softmax(logits, dim=1).cpu().numpy()
                calibrated_probs = np.vstack(
                    [iso.predict(probabilities[:, c]) for c, iso in zip(self.classes_, self.calibrator, strict=False)],
                ).T

                calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
                return tensor(calibrated_probs, device=self.device)

            # Case Binary Model
            scores = self._get_binary_scores(logits)
            probabilities = sigmoid(scores).cpu().numpy()
            calibrated_probs = self.calibrator[0].predict(probabilities)
            calibrated_probs = np.stack([1 - calibrated_probs, calibrated_probs], axis=1)
            return tensor(calibrated_probs, device=self.device, dtype=logits.dtype)

    def _extract_logits_and_labels(self, dataset: DataLoader) -> tuple[Tensor, Tensor]:
        """Returns the logits and labels for a dataset as a tuple (logits, labels)."""
        self.model.eval()

        logits = []
        labels = []

        with no_grad():
            for x, y in dataset:
                inpt = x.to(self.device)
                output = self.model(inpt)
                logits.append(output.cpu())
                labels.append(y.cpu())

        return (cat(logits), cat(labels))

    def _get_binary_scores(self, logits: Tensor) -> Tensor:
        if logits.ndim == 1:
            return logits

        if logits.shape[1] == 1:
            return logits.squeeze()

        if logits.shape[1] == 2:
            return logits[:, 1]

        value_error_str = f"Binary Model should output shape (N,2), (N,1), (N,), not {tuple(logits.shape)}"
        raise ValueError(value_error_str)
