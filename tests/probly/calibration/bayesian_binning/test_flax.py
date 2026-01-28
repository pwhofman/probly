"""Tests for flax Bayesian Binning into Quantiles (BBQ)."""

from __future__ import annotations

import pytest

from probly.calibration.bayesian_binning.flax import BayesianBinningQuantiles

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


class TestBayesianBinningQuantiles:
    """Test class for bayesian binnig quantiles in torch."""

    def test_fitted_check(self) -> None:
        """Ensure predict raises error if called before fit."""
        calibrator = BayesianBinningQuantiles(max_bins=5)
        test_predictions = jnp.array([0.1, 0.4, 0.6, 0, 8])
        with pytest.raises(RuntimeError, match="Calibrator must be fitted before prediction"):
            calibrator.predict(test_predictions)

    def test_tensors_shape_mismatch(self) -> None:
        """Ensure fit raises error on mismatched tensor lengths."""
        calibrator = BayesianBinningQuantiles(max_bins=5)
        calibration_set = jnp.array([0.1, 0.4, 0.6, 0.8])
        truth_labels = jnp.array([0, 1, 0])
        with pytest.raises(ValueError, match="Calibration_set and truth_labels must have same length"):
            calibrator.fit(calibration_set, truth_labels)

    def test_empty_calibration_set(self) -> None:
        """Ensure fit raises error on empty calibration set."""
        calibrator = BayesianBinningQuantiles(max_bins=5)
        calibration_set = jnp.array([])
        truth_labels = jnp.array([])
        with pytest.raises(ValueError, match="Calibration_set cannot be empty"):
            calibrator.fit(calibration_set, truth_labels)

    def test_calibration_between_0_and_1(self) -> None:
        """Ensure calibrated probabilities are valid probabilities."""
        calibrator = BayesianBinningQuantiles(max_bins=5)
        calibration_set = jnp.array(
            [0.1, 0.4, 0.6, 0.8, 0.2, 0.9, 0.3, 0.5],
        )
        truth_labels = jnp.array([0, 0, 1, 1, 0, 1, 0, 1])
        calibrator.fit(calibration_set, truth_labels)
        test_predictions = jnp.array([0.15, 0.45, 0.65, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert bool(
            jnp.all((calibrated_probs >= 0.0) & (calibrated_probs <= 1.0)),
        ), "calibrated probabilities must be between 0 and 1"

    def test_output_shape_matches_input(self) -> None:
        """Ensure predict output shape matches input shape."""
        calibrator = BayesianBinningQuantiles(max_bins=5)
        calibration_set = jnp.array(
            [0.1, 0.4, 0.6, 0.8, 0.2, 0.9, 0.3, 0.5],
        )
        truth_labels = jnp.array([0, 0, 1, 1, 0, 1, 0, 1])
        calibrator.fit(calibration_set, truth_labels)
        test_predictions = jnp.array([0.15, 0.45, 0.65, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert calibrated_probs.shape == test_predictions.shape

    def test_default_behaviour_simple_case(self) -> None:
        """Higher predictions should yield higher calibrated probabilities."""
        calibrator = BayesianBinningQuantiles(max_bins=3)
        calibration_set = jnp.array([0.1, 0.2, 0.8, 0.9])
        truth_labels = jnp.array([0, 0, 1, 1])
        calibrator.fit(calibration_set, truth_labels)
        test_predictions = jnp.array([0.15, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert calibrated_probs[0] < calibrated_probs[1]

    def test_predict_is_deterministic(self) -> None:
        """Ensure repeated predictions give identical results."""
        calibrator = BayesianBinningQuantiles(max_bins=4)
        calibration_set = jnp.linspace(0.1, 0.9, 10)
        truth_labels = jnp.array([0, 1] * 5)
        calibrator.fit(calibration_set, truth_labels)
        predictions = jnp.array([0.25, 0.5, 0.75])
        out1 = calibrator.predict(predictions)
        out2 = calibrator.predict(predictions)

        assert jnp.allclose(out1, out2)
