"""The BBQ Calibrator with Flax."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax.numpy as jnp
from jax.scipy.special import betaln

from probly.calibration.bayesian_binning.common import register_bayesian_binning_factory
from probly.calibration.template import CalibratorBaseFlax

if TYPE_CHECKING:
    from jax import Array


class BayesianBinningQuantiles(CalibratorBaseFlax):
    """Calibrator using Bayesian Binning into Quantiles (BBQ)."""

    def __init__(self, max_bins: int = 10) -> None:
        """Initialize the class."""
        self.max_bins = max_bins

        self.bin_edges: list[Array] = []
        self.system_bin_probs: list[Array] = []

        self.system_scores: list[Array] = []
        self.system_weights: list[float] = []

        self.is_fitted = False

    def fit(self, calibration_set: Array, truth_labels: Array) -> BayesianBinningQuantiles:
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
            edges = jnp.quantile(
                calibration_set,
                jnp.linspace(0.0, 1.0, num_bins + 1),
            )
            edges = edges.at[0].set(0.0)
            edges = edges.at[-1].set(1.0)
            self.bin_edges.append(edges)

            # Assign samples to bins
            bin_ids = jnp.digitize(calibration_set, edges) - 1
            bin_ids = jnp.clip(bin_ids, 0, num_bins - 1)

            # Count samples per bin
            bin_counts = jnp.zeros(num_bins, dtype=jnp.int32)
            bin_positives = jnp.zeros(num_bins, dtype=jnp.int32)

            for i in range(calibration_set.shape[0]):
                b = bin_ids[i]
                bin_counts = bin_counts.at[b].add(1)
                bin_positives = bin_positives.at[b].add(truth_labels[i])

            # Bayesian smoothed bin probabilities
            bin_probs = jnp.where(
                bin_counts > 0,
                (bin_positives + 1.0) / (bin_counts + 2.0),
                0.5,
            )
            self.system_bin_probs.append(bin_probs)

            # Bin marginal likelihoods (log-space)
            log_bin_scores = betaln(
                bin_positives + 1.0,
                bin_counts - bin_positives + 1.0,
            )

            system_log_score = jnp.sum(log_bin_scores)
            system_score = jnp.exp(system_log_score)
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

        calibrated = jnp.zeros_like(predictions, dtype=jnp.float32)

        for i in range(predictions.shape[0]):
            pred = predictions[i]
            calibrated_prob = jnp.array(0.0)

            for sys_idx, edges in enumerate(self.bin_edges):
                bin_probs = self.system_bin_probs[sys_idx]
                weight = self.system_weights[sys_idx]

                bin_idx = jnp.digitize(pred, edges) - 1
                bin_idx = jnp.clip(bin_idx, 0, bin_probs.shape[0] - 1)

                calibrated_prob = calibrated_prob + weight * bin_probs[bin_idx]

            calibrated = calibrated.at[i].set(calibrated_prob)

        return calibrated


@register_bayesian_binning_factory(nnx.Module)
def _(_base: nnx.Module, _device: object) -> type[BayesianBinningQuantiles]:
    return BayesianBinningQuantiles
