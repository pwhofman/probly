"""The BBQ Calibrator with Flax."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.special import betaln

if TYPE_CHECKING:
    from jax import Array


class BayesianBinningQuantilesFlax:
    """Calibrator using Bayesian Binning into Quantiles (BBQ)."""

    def __init__(self, max_bins: int = 10) -> None:
        """Initialize the class."""
        self.max_bins = max_bins
        self.bin_edges: list[Array] = []
        self.system_bin_probs: list[Array] = []
        self.system_scores: list[Array] = []
        self.system_weights: list[float] = []
        self.is_fitted = False

    def fit(self, calibration_set: Array, truth_labels: Array) -> BayesianBinningQuantilesFlax:
        """Fit the BBQ calibrator."""
        if calibration_set.shape[0] != truth_labels.shape[0]:
            msg = "Calibration_set and truth_labels must have same length"
            raise ValueError(msg)
        if calibration_set.shape[0] == 0:
            msg = "Calibration_set cannot be empty"
            raise ValueError(msg)

        self.system_bin_probs = []
        self.system_scores = []
        self.bin_edges = []

        for num_bins in range(2, self.max_bins + 1):
            # Quantile-based bin edges
            edges = jnp.quantile(calibration_set, jnp.linspace(0.0, 1.0, num_bins + 1))
            edges = edges.at[0].set(0.0)
            edges = edges.at[-1].set(1.0)
            self.bin_edges.append(edges)

            # Vectorized bin assignment
            bin_ids = jnp.digitize(calibration_set, edges) - 1
            bin_ids = jnp.clip(bin_ids, 0, num_bins - 1)

            # Vectorized bin counts and positives
            bin_counts = jnp.bincount(bin_ids, length=num_bins)
            bin_positives = jnp.bincount(bin_ids, weights=truth_labels.astype(jnp.float32), length=num_bins)

            # Bayesian smoothed bin probabilities
            bin_probs = jnp.where(bin_counts > 0, (bin_positives + 1.0) / (bin_counts + 2.0), 0.5)
            self.system_bin_probs.append(bin_probs)

            # Bin marginal likelihoods (log-space)
            log_bin_scores = betaln(bin_positives + 1.0, bin_counts - bin_positives + 1.0)
            system_score = jnp.exp(jnp.sum(log_bin_scores))
            self.system_scores.append(system_score)

        # Normalize weights
        scores = jnp.array(self.system_scores)
        self.system_weights = (scores / scores.sum()).tolist()

        self.is_fitted = True
        return self

    def predict(self, predictions: Array) -> Array:
        """Return calibrated probabilities for input predictions."""
        if not self.is_fitted:
            msg = "Calibrator must be fitted before prediction"
            raise RuntimeError(msg)

        predictions = predictions[:, None]
        calibrated_probs = jnp.zeros_like(predictions, dtype=jnp.float32)

        for edges, bin_probs, weight in zip(self.bin_edges, self.system_bin_probs, self.system_weights, strict=False):
            bin_indices = jnp.digitize(predictions, edges) - 1
            bin_indices = jnp.clip(bin_indices, 0, bin_probs.shape[0] - 1)

            probs = bin_probs[bin_indices]
            calibrated_probs += weight * probs

        return calibrated_probs.ravel()
