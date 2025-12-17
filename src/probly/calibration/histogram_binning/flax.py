"""The Histogram Binning Calibrator with Flax."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import linen as nn
import jax.numpy as jnp

from probly.calibration.histogram_binning.common import register_histogram_factory
from probly.calibration.template import CalibratorBaseFlax

if TYPE_CHECKING:
    from jax import Array


# histogram binnig
class HistogramBinningFlax(CalibratorBaseFlax):
    """Calibrator that uses histogram binning."""

    def __init__(self, base_model: nn.Module, params: dict, n_bins: int = 10) -> None:
        """Create a histogram binning calibrator.

        Args:
            base_model: The base Flax model
            params: Model parameters
            n_bins: Number of bins for histogram
        """
        super().__init__(base_model, params)
        self.n_bins = n_bins
        self.bin_start = 0.0
        self.bin_width = 0.0
        self.bin_probs: Array | None = None

    def fit(self, calibration_set: Array, truth_labels: Array) -> HistogramBinningFlax:
        """Fit the histogram binning calibrator.

        Args:
            calibration_set: Predictions from the model
            truth_labels: Ground truth labels

        Returns:
            Self with fitted parameters
        """
        min_pre = float(jnp.min(calibration_set))
        max_pre = float(jnp.max(calibration_set))
        bin_width = (max_pre - min_pre) / self.n_bins

        if max_pre == min_pre:
            bin_width = 1.0

        bin_counts = jnp.zeros(self.n_bins, dtype=jnp.int32)
        bin_positives = jnp.zeros(self.n_bins, dtype=jnp.int32)

        # Vectorized binning (more efficient than loop in JAX)
        bin_indices = jnp.floor((calibration_set - min_pre) / bin_width).astype(jnp.int32)
        bin_indices = jnp.clip(bin_indices, 0, self.n_bins - 1)

        # Count samples per bin
        bin_counts = jnp.bincount(bin_indices, length=self.n_bins)

        # Count positive samples per bin
        bin_positives = jnp.bincount(
            bin_indices,
            weights=truth_labels.astype(jnp.float32),
            length=self.n_bins,
        ).astype(jnp.int32)

        # Calculate bin probabilities
        # Use where to avoid division by zero
        self.bin_probs = jnp.where(
            bin_counts > 0,
            bin_positives.astype(jnp.float32) / bin_counts.astype(jnp.float32),
            0.0,
        )

        self.bin_start = min_pre
        self.bin_width = bin_width
        return self

    def predict(self, predictions: Array) -> Array:
        """Return calibrated probabilities for input predictions.

        Args:
            predictions: Model predictions to calibrate

        Returns:
            Calibrated probabilities
        """
        if self.bin_probs is None:
            msg = "HistogramBinning is not fitted"
            raise RuntimeError(msg)

        # Vectorized prediction (more efficient than loop)
        bin_indices = jnp.floor((predictions - self.bin_start) / self.bin_width).astype(jnp.int32)
        bin_indices = jnp.clip(bin_indices, 0, self.n_bins - 1)

        # Look up calibrated probabilities
        calibrated = self.bin_probs[bin_indices]

        return calibrated


register_histogram_factory(nn.Module)(HistogramBinningFlax)
