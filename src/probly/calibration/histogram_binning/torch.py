"""The Histogram Binning Calibrator with Torch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.histogram_binning.common import register_histogram_factory
from probly.calibration.template import CalibratorBaseTorch


@register_histogram_factory(nn.Module)
class HistogramBinning(CalibratorBaseTorch):
    """Calibrator that uses histogram binning."""

    def __init__(self, n_bins: int = 10) -> None:
        """Create a histogram binning calibrator."""
        self.n_bins = n_bins
        self.bin_start = 0.0
        self.bin_width = 0.0
        self.is_fitted = False
        self.bin_probs: Tensor | None = None

    def fit(self, calibration_set: Tensor, truth_labels: Tensor) -> HistogramBinning:
        """Fit the histogram binning calibrator."""
        if calibration_set.shape[0] != truth_labels.shape[0]:
            msg = "calibration_set and truth_labels must have the same length"
            raise ValueError(msg)

        if calibration_set.shape[0] == 0:
            msg = "calibration_set must not be empty"
            raise ValueError(msg)

        min_pre = calibration_set.min().item()
        max_pre = calibration_set.max().item()
        bin_width = (max_pre - min_pre) / self.n_bins

        if max_pre == min_pre:
            bin_width = 1.0

        bin_counts = torch.zeros(self.n_bins, dtype=torch.int64)
        bin_positives = torch.zeros(self.n_bins, dtype=torch.int64)

        for pred, label in zip(calibration_set, truth_labels, strict=False):
            bin_id = int((pred.item() - min_pre) / bin_width)
            # Test for the case where bin id is equal to n_bins, making it out of bounds MIGHT REMOVE LATER
            bin_id = max(0, min(bin_id, self.n_bins - 1))
            if bin_id == self.n_bins:
                bin_id = self.n_bins - 1
            bin_counts[bin_id] += 1
            bin_positives[bin_id] += int(label.item())

        self.bin_probs = torch.zeros(self.n_bins)
        for i in range(self.n_bins):
            if bin_counts[i] > 0:
                self.bin_probs[i] = bin_positives[i].float() / bin_counts[i].float()
            else:
                self.bin_probs[i] = 0.0

        self.bin_start = min_pre
        self.bin_width = bin_width
        self.is_fitted = True
        return self

    def predict(self, predictions: Tensor) -> Tensor:
        """Return calibrated probabilities for input predictions."""
        if not self.is_fitted:
            msg = "Calibrator must be fitted before Calibration"
            raise ValueError(msg)

        calibrated = []

        for pred in predictions:
            bin_id = int((pred.item() - self.bin_start) / self.bin_width)
            if bin_id == self.n_bins:
                bin_id -= 1

            # Temporary fix for mypy issue: self.bin_probs could theoretically be None
            if self.bin_probs is None:
                msg = "HistogramBinning is not fitted"
                raise RuntimeError(msg)
            calibrated.append(self.bin_probs[bin_id])

        return Tensor(calibrated)
