"""The BBQ Calibrator with Torch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.bayesian_binning.common import register_bayesian_binning_factory

from .utils import betaln


class BayesianBinningQuantiles:
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

    def fit(self, calibration_set: Tensor, truth_labels: Tensor) -> BayesianBinningQuantiles:
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
            # Quantile-based bin edges - bins have relatively equal numbers of elements
            edges = torch.quantile(calibration_set, torch.linspace(0, 1, num_bins + 1))
            edges[0] = 0.0
            edges[-1] = 1.0
            self.bin_edges.append(edges)

            # Compute n and k for each bin
            bin_counts = torch.zeros(num_bins, dtype=torch.int64)
            bin_positives = torch.zeros(num_bins, dtype=torch.int64)

            # Assign each sample to a bin
            bin_ids = torch.bucketize(calibration_set, edges) - 1
            bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)

            for idx, bin_id in enumerate(bin_ids):
                bin_counts[bin_id] += 1
                bin_positives[bin_id] += truth_labels[idx].item()

            # Bayesian smoothed probabilities
            bin_probs = torch.zeros(num_bins, dtype=torch.float32)
            for i in range(num_bins):
                bin_probs[i] = (bin_positives[i].float() + 1.0) / (bin_counts[i].float() + 2.0)

            self.system_bin_probs.append(bin_probs)

            # Compute bin scores
            log_bin_scores = torch.zeros(num_bins)
            for i in range(num_bins):
                k = bin_positives[i].item()
                n = bin_counts[i].item()

                # Convert to tensor with float dtype (and device if needed)
                a = torch.tensor(k + 1, dtype=torch.float32)
                b = torch.tensor(n - k + 1, dtype=torch.float32)

                log_bin_scores[i] = betaln(a, b) - betaln(torch.tensor(1.0), torch.tensor(1.0))

            # System score = product of bin scores (in log-space)
            system_log_score = log_bin_scores.sum()
            self.system_scores.append(system_log_score)

        # Normalize system scores to weights
        log_scores = torch.stack(self.system_scores)
        weights = torch.softmax(log_scores, dim=0)
        self.system_weights = weights.tolist()
        self.is_fitted = True
        return self

    def predict(self, predictions: Tensor) -> Tensor:
        """Return calibrated probabilities for input predictions."""
        if not self.is_fitted:
            msg = "Calibrator must be fitted before prediction"
            raise RuntimeError(msg)

        calibrated = torch.zeros_like(predictions, dtype=torch.float32)

        for i, pred in enumerate(predictions):
            calibrated_prob = torch.zeros((), dtype=torch.float32, device=pred.device)
            for sys_idx, edges in enumerate(self.bin_edges):
                bin_probs = self.system_bin_probs[sys_idx]
                weight = self.system_weights[sys_idx]
                bin_idx = torch.bucketize(pred, edges) - 1
                bin_idx = torch.clamp(bin_idx, 0, len(bin_probs) - 1)
                calibrated_prob += weight * bin_probs[bin_idx]
            calibrated[i] = calibrated_prob

        return calibrated


@register_bayesian_binning_factory(nn.Module)
def _(_base: nn.Module, _device: object) -> type[BayesianBinningQuantiles]:
    return BayesianBinningQuantiles
