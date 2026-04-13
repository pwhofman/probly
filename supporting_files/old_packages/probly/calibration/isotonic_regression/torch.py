"""Torch Implementations for Isotonic Regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.isotonic import IsotonicRegression
from torch import Tensor, as_tensor, cat, nn, no_grad, sigmoid, softmax, unique

from probly.calibration.isotonic_regression import common

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class IsotonicRegressionCalibrator:
    """Class for the isotonic regression calibration."""

    def __init__(self, base_model: nn.Module, use_logits: bool) -> None:
        """Set up the wrapper.

        Args:
            base_model: The base model to calibrate
            use_logits: A switch allowing to return logits or probabilities

        """
        self.model = base_model.to("cpu")
        self.use_logits = use_logits
        self.calibrator: list[IsotonicRegression] = []

    def fit(self, calibration_set: DataLoader) -> None:
        """Fit the regression function to the model outputs.

        Args:
            calibration_set: The set that should be used for the calibration

        """
        self.calibrator = []

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

            if self.use_logits:
                iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso_reg.fit(scores.cpu().numpy(), labels_bin.cpu().numpy())
                self.calibrator = [iso_reg]

            else:
                probabilities = sigmoid(scores).cpu()

                iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso_reg.fit(probabilities.numpy(), labels_bin.cpu().numpy())
                self.calibrator = [iso_reg]

        elif num_classes > 2:
            if self.use_logits:
                for c in self.classes_:
                    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    scores_c = logits_cal[:, c].cpu().numpy()
                    targets_c = (labels_cal == c).cpu().numpy().astype(np.int64)
                    iso_reg.fit(scores_c, targets_c)
                    self.calibrator.append(iso_reg)

            else:
                probabilities = softmax(logits_cal, dim=1).cpu().numpy()

                for c in self.classes_:
                    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso_reg.fit(probabilities[:, c], (labels_cal == c).cpu().numpy())
                    self.calibrator.append(iso_reg)

    def predict(self, x: Tensor) -> Tensor:
        """Make calibrated predictions on the input x.

        Args:
            x: The input for the model to make predictions on

        Returns:
            calibrated_probs: The calibrated probabilities for the prediction

        """
        x = x.to("cpu")
        self.model.eval()

        with no_grad():
            logits = self.model(x)

            if len(self.calibrator) == 0:
                calibrator_none_message = "Calibrator has not been fitted yet. Call fit method first."
                raise RuntimeError(calibrator_none_message)

            # Case Multiclass model
            if len(self.calibrator) >= 2:
                s = logits.cpu().numpy() if self.use_logits else softmax(logits, dim=1).cpu().numpy()

                calibrated = np.vstack(
                    [iso.predict(s[:, c]) for c, iso in zip(self.classes_, self.calibrator, strict=False)],
                ).T

                calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
                return as_tensor(calibrated, device=x.device, dtype=logits.dtype)

            # Case Binary Model
            scores = self._get_binary_scores(logits)
            s = scores.cpu().numpy() if self.use_logits else sigmoid(scores).cpu().numpy()

            calibrated = self.calibrator[0].predict(s)
            calibrated = np.stack([1 - calibrated, calibrated], axis=1)
            return as_tensor(calibrated, device=x.device, dtype=logits.dtype)

    def _extract_logits_and_labels(self, dataset: DataLoader) -> tuple[Tensor, Tensor]:
        """Returns the logits and labels for a dataset as a tuple (logits, labels)."""
        self.model.eval()

        logits = []
        labels = []

        with no_grad():
            for x, y in dataset:
                inpt = x.to("cpu")
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


@common.register_isotonic_factory(nn.Module)
def _(_base_model: nn.Module, _use_logits: bool) -> type[IsotonicRegressionCalibrator]:
    return IsotonicRegressionCalibrator
