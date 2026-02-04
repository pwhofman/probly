"""The Histogram Binning Calibrator with Torch."""

from __future__ import annotations

import torch
from torch import Tensor


class HistogramBinningTorch:
    """Calibrator that uses histogram binning."""

    def __init__(self, n_bins: int = 10) -> None:
        """Create a histogram binning calibrator."""
        self.n_bins = n_bins
        self.bin_start = 0.0
        self.bin_width = 0.0
        self.is_fitted = False
        self.bin_probs: Tensor | None = None

    def fit(self, calibration_set: Tensor, truth_labels: Tensor) -> HistogramBinningTorch:
        """Fit the histogram binning calibrator."""
        if calibration_set.shape[0] != truth_labels.shape[0]:
            msg = "calibration_set and truth_labels must have the same length"
            raise ValueError(msg)

        if calibration_set.numel() == 0:
            msg = "calibration_set must not be empty"
            raise ValueError(msg)

        min_pre = calibration_set.min()
        max_pre = calibration_set.max()

        bin_width = (max_pre - min_pre) / self.n_bins
        if bin_width == 0:
            bin_width = torch.tensor(1.0, device=calibration_set.device)

        bin_edges = min_pre + bin_width * torch.arange(
            1,
            self.n_bins,
            device=calibration_set.device,
        )

        bin_ids = torch.bucketize(calibration_set, bin_edges)

        bin_ids = torch.clamp(bin_ids, 0, self.n_bins - 1)

        bin_counts = torch.bincount(bin_ids, minlength=self.n_bins)

        bin_positives = torch.bincount(
            bin_ids,
            weights=truth_labels.to(dtype=torch.float),
            minlength=self.n_bins,
        )

        self.bin_probs = torch.zeros(self.n_bins, device=calibration_set.device)
        nonzero = bin_counts > 0
        self.bin_probs[nonzero] = bin_positives[nonzero] / bin_counts[nonzero]

        self.bin_start = float(min_pre)
        self.bin_width = float(bin_width)
        self.is_fitted = True
        return self

    def predict(self, predictions: Tensor) -> Tensor:
        """Return calibrated probabilities for input predictions."""
        if not self.is_fitted or self.bin_probs is None:
            msg = "Calibrator must be fitted before Calibration"
            raise ValueError(msg)

        bin_ids = ((predictions - self.bin_start) / self.bin_width).long()
        bin_ids = torch.clamp(bin_ids, 0, self.n_bins - 1)

        return self.bin_probs[bin_ids]
