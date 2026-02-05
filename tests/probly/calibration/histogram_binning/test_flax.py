"""Tests for flax calibration methods using histogram binning."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from probly.calibration.histogram_binning.flax import HistogramBinningFlax


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_fitted_check(self) -> None:
        """Tests if the calibrator raises an error when predict is called before fit."""
        calibrator = HistogramBinningFlax(n_bins=5)

        test_predictions = jnp.array([0.1, 0.4, 0.6, 0.8])

        with pytest.raises(ValueError, match="Calibrator must be fitted before Calibration"):
            calibrator.predict(test_predictions)

    def test_tensors_shape_mismatch(self) -> None:
        """Tests if the calibrator raises an error when fit is called with mismatched tensor shapes."""
        calibrator = HistogramBinningFlax(n_bins=5)

        calibration_set = jnp.array([0.1, 0.4, 0.6, 0.8])
        truth_labels = jnp.array([0, 1, 1])

        with pytest.raises(ValueError, match="calibration_set and truth_labels must have the same length"):
            calibrator.fit(calibration_set, truth_labels)

    def test_empty_calibration_set(self) -> None:
        """Tests if the calibrator raises an error when fit is called with an empty calibration set."""
        calibrator = HistogramBinningFlax(n_bins=5)

        calibration_set = jnp.array([])
        truth_labels = jnp.array([])

        with pytest.raises(ValueError, match="calibration_set must not be empty"):
            calibrator.fit(calibration_set, truth_labels)

    def test_calibration_between_0_and_1(self) -> None:
        """Tests if the calibrator produces calibrated probabilities between 0 and 1."""
        calibrator = HistogramBinningFlax(n_bins=5)

        calibration_set = jnp.array([0.1, 0.4, 0.6, 0.8, 0.2, 0.9, 0.3, 0.5])
        truth_labels = jnp.array([0, 0, 1, 1, 0, 1, 0, 1])

        calibrator.fit(calibration_set, truth_labels)

        test_predictions = jnp.array([0.15, 0.45, 0.65, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert jnp.all((calibrated_probs >= 0) & (calibrated_probs <= 1)), (
            "Calibrated probabilities are not between 0 and 1."
        )

    def test_calibration_default_behaviour(self) -> None:
        """Tests the default behaviour of the histogram binning calibrator."""
        calibrator = HistogramBinningFlax(n_bins=4)

        calibration_set = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        truth_labels = jnp.array([0, 0, 1, 1, 0, 1, 1, 1])

        calibrator.fit(calibration_set, truth_labels)

        test_predictions = jnp.array([0.15, 0.35, 0.55, 0.75])
        calibrated_probs = calibrator.predict(test_predictions)

        expected_probs = jnp.array([0.0, 1.0, 0.5, 1.0])
        assert jnp.allclose(calibrated_probs, expected_probs), "Calibrated probabilities do not match expected values"
