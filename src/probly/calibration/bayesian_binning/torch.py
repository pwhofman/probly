"""The BBQ Calibrator with Torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class BayesianBinningQuantilesTorch:
    """Calibrator using Bayesian Binning into Quantiles (BBQ)."""

    def __init__(self, max_bins: int = 10) -> None:
        """Pass a dummy model if you don't need one."""
        self.max_bins = max_bins
        self.bin_edges: list[Tensor] = []
        self.system_bin_probs: list[Tensor] = []
        self.system_scores: list[Tensor] = []
        self.system_weights: list[float] = []
        self.is_fitted = False
        self.system_log_scores: list[Tensor] = []

    def fit(self, calibration_set: torch.Tensor, truth_labels: Tensor) -> BayesianBinningQuantilesTorch:
        """Fit the BBQ calibrator."""
        if calibration_set.shape[0] != truth_labels.shape[0]:
            msg = "calibration_set and truth_labels must have same length"
            raise ValueError(msg)
        if calibration_set.shape[0] == 0:
            msg = "calibration_set cannot be empty"
            raise ValueError(msg)

        self.system_bin_probs = []
        self.system_scores = []
        self.bin_edges = []

        for num_bins in range(2, self.max_bins + 1):
            edges = torch.quantile(calibration_set, torch.linspace(0.0, 1.0, num_bins + 1))
            edges[0] = 0.0
            edges[-1] = 1.0
            self.bin_edges.append(edges)

            bin_ids = torch.bucketize(calibration_set, edges, right=False) - 1
            bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)

            bin_counts = torch.bincount(bin_ids, minlength=num_bins)
            bin_positives = torch.bincount(bin_ids, weights=truth_labels.float(), minlength=num_bins)

            bin_probs = torch.where(
                bin_counts > 0,
                (bin_positives + 1.0) / (bin_counts + 2.0),
                torch.tensor(0.5, dtype=torch.float32),
            )
            self.system_bin_probs.append(bin_probs)

            log_bin_scores = (
                torch.lgamma(bin_positives + 1.0)
                + torch.lgamma(bin_counts - bin_positives + 1.0)
                - torch.lgamma(bin_counts + 2.0)
            )
            system_score = torch.exp(torch.sum(log_bin_scores))
            self.system_scores.append(system_score)

        scores = torch.stack(self.system_scores)
        self.system_weights = (scores / scores.sum()).tolist()

        self.is_fitted = True
        return self

    def predict(self, predictions: Tensor) -> Tensor:
        """Return calibrated probabilities for input predictions."""
        if not self.is_fitted:
            msg = "Calibrator must be fitted before prediction"
            raise RuntimeError(msg)

        calibrated = torch.zeros_like(predictions, dtype=torch.float32)

        for edges, bin_probs, weight in zip(self.bin_edges, self.system_bin_probs, self.system_weights, strict=False):
            bin_indices = torch.bucketize(predictions, edges, right=False) - 1
            bin_indices = torch.clamp(bin_indices, 0, bin_probs.shape[0] - 1)

            probs = bin_probs[bin_indices]
            calibrated += weight * probs

        return calibrated

    def _betaln(self, a: Tensor, b: Tensor) -> Tensor:
        """Natural log of the Beta Function."""
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
