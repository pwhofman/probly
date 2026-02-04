"""The Histogram Binning Calibrator with Flax."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


class HistogramBinningFlax:
    """Calibrator that uses histogram binning."""

    def __init__(self, n_bins: int = 10) -> None:
        """Create a histogram binning calibrator.

        Args:
            base_model: The base Flax model
            params: Model parameters
            n_bins: Number of bins for histogram
        """
        self.n_bins = n_bins
        self.bin_start = 0.0
        self.bin_width = 0.0
        self.is_fitted = False
        self.bin_probs: Array | None = None

    def fit(self, calibration_set: Array, truth_labels: Array) -> HistogramBinningFlax:
        """Fit the histogram binning calibrator."""
        if calibration_set.shape[0] != truth_labels.shape[0]:
            msg = "calibration_set and truth_labels must have the same length"
            raise ValueError(msg)

        if calibration_set.shape[0] == 0:
            msg = "calibration_set must not be empty"
            raise ValueError(msg)

        min_pre = float(jnp.min(calibration_set))
        max_pre = float(jnp.max(calibration_set))
        bin_width = (max_pre - min_pre) / self.n_bins

        if max_pre == min_pre:
            bin_width = 1.0

        bin_counts = jnp.zeros(self.n_bins, dtype=jnp.int32)
        bin_positives = jnp.zeros(self.n_bins, dtype=jnp.int32)

        bin_indices = jnp.floor((calibration_set - min_pre) / bin_width).astype(jnp.int32)
        bin_indices = jnp.clip(bin_indices, 0, self.n_bins - 1)

        bin_counts = jnp.bincount(bin_indices, length=self.n_bins)

        bin_positives = jnp.bincount(
            bin_indices,
            weights=truth_labels.astype(jnp.float32),
            length=self.n_bins,
        ).astype(jnp.int32)

        self.bin_probs = jnp.where(
            bin_counts > 0,
            bin_positives.astype(jnp.float32) / bin_counts.astype(jnp.float32),
            0.0,
        )

        self.bin_start = min_pre
        self.bin_width = bin_width
        self.is_fitted = True
        return self

    def predict(self, predictions: Array) -> Array:
        """Return calibrated probabilities for input predictions."""
        if not self.is_fitted:
            msg = "Calibrator must be fitted before Calibration"
            raise ValueError(msg)
        if self.bin_probs is None:
            msg = "HistogramBinning is not fitted"
            raise RuntimeError(msg)

        bin_indices = jnp.floor((predictions - self.bin_start) / self.bin_width).astype(jnp.int32)
        bin_indices = jnp.clip(bin_indices, 0, self.n_bins - 1)

        calibrated = self.bin_probs[bin_indices]

        return calibrated
