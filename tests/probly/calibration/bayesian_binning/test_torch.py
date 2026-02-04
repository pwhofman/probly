"""Tests for torch Bayesian Binning into Quantiles (BBQ)."""

from __future__ import annotations

import pytest

from probly.calibration.bayesian_binning.torch import BayesianBinningQuantilesTorch

torch = pytest.importorskip("torch")


class TestBayesianBinningQuantilesTorch:
    """Test class for bayesian binnig quantiles in torch."""

    def test_fitted_check(self) -> None:
        """Ensure predict raises error if called before fit."""
        calibrator = BayesianBinningQuantilesTorch(max_bins=5)
        test_predictions = torch.tensor([0.1, 0.4, 0.6, 0, 8])
        with pytest.raises(RuntimeError, match="Calibrator must be fitted before prediction"):
            calibrator.predict(test_predictions)

    def test_tensors_shape_mismatch(self) -> None:
        """Ensure fit raises error on mismatched tensor lengths."""
        calibrator = BayesianBinningQuantilesTorch(max_bins=5)
        calibration_set = torch.tensor([0.1, 0.4, 0.6, 0.8])
        truth_labels = torch.tensor([0, 1, 0])
        with pytest.raises(ValueError, match="calibration_set and truth_labels must have same length"):
            calibrator.fit(calibration_set, truth_labels)

    def test_empty_calibration_set(self) -> None:
        """Ensure fit raises error on empty calibration set."""
        calibrator = BayesianBinningQuantilesTorch(max_bins=5)
        calibration_set = torch.tensor([])
        truth_labels = torch.tensor([])
        with pytest.raises(ValueError, match="calibration_set cannot be empty"):
            calibrator.fit(calibration_set, truth_labels)

    def test_calibration_between_0_and_1(self) -> None:
        """Ensure calibrated probabilities are valid probabilities."""
        calibrator = BayesianBinningQuantilesTorch(max_bins=5)
        calibration_set = torch.tensor(
            [0.1, 0.4, 0.6, 0.8, 0.2, 0.9, 0.3, 0.5],
        )
        truth_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
        calibrator.fit(calibration_set, truth_labels)
        test_predictions = torch.tensor([0.15, 0.45, 0.65, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert torch.all(
            (calibrated_probs >= 0.0) & (calibrated_probs <= 1.0),
        )

    def test_output_shape_matches_input(self) -> None:
        """Ensure predict output shape matches input shape."""
        calibrator = BayesianBinningQuantilesTorch(max_bins=5)
        calibration_set = torch.tensor(
            [0.1, 0.4, 0.6, 0.8, 0.2, 0.9, 0.3, 0.5],
        )
        truth_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
        calibrator.fit(calibration_set, truth_labels)
        test_predictions = torch.tensor([0.15, 0.45, 0.65, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert calibrated_probs.shape == test_predictions.shape

    def test_default_behaviour_simple_case(self) -> None:
        """Higher predictions should yield higher calibrated probabilities."""
        calibrator = BayesianBinningQuantilesTorch(max_bins=3)
        calibration_set = torch.tensor([0.1, 0.2, 0.8, 0.9])
        truth_labels = torch.tensor([0, 0, 1, 1])
        calibrator.fit(calibration_set, truth_labels)
        test_predictions = torch.tensor([0.15, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert calibrated_probs[0] < calibrated_probs[1], (
            "Higher input predicitions should result in highor calibrated probabilities"
        )
