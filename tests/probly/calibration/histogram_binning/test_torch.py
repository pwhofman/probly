"""Tests for torch calibration methods using histogram binning."""

from __future__ import annotations

import pytest

from probly.calibration.histogram_binning.torch import HistogramBinningTorch

torch = pytest.importorskip("torch")


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_fitted_check(self) -> None:
        """Tests if the calibrator raises an error when predict is called before fit.

        It performs a check to ensure that the calibrator raises a ValueError when the predict method is called before
        the calibrator has been fitted.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the dropout layer is not
            inserted correctly after linear layers.
        """
        calibrator = HistogramBinningTorch(n_bins=5)

        test_predictions = torch.tensor([0.1, 0.4, 0.6, 0.8])

        with pytest.raises(ValueError, match="Calibrator must be fitted before Calibration"):
            calibrator.predict(test_predictions)

    def test_tensors_shape_mismatch(self) -> None:
        """Tests if the calibrator raises an error when fit is called with mismatched tensor shapes.

        It performs a check to ensure that the calibrator raises a ValueError when the fit method is called with
        calibration_set and truth_labels tensors of different lengths.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the dropout layer is not
            inserted correctly after linear layers.
        """
        calibrator = HistogramBinningTorch(n_bins=5)

        calibration_set = torch.tensor([0.1, 0.4, 0.6, 0.8])
        truth_labels = torch.tensor([0, 1, 1])

        with pytest.raises(ValueError, match="calibration_set and truth_labels must have the same length"):
            calibrator.fit(calibration_set, truth_labels)

    def test_empty_calibration_set(self) -> None:
        """Tests if the calibrator raises an error when fit is called with an empty calibration set.

        It performs a check to ensure that the calibrator raises a ValueError when the fit method is called with
        an empty calibration_set tensor.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the dropout layer is not
            inserted correctly after linear layers.
        """
        calibrator = HistogramBinningTorch(n_bins=5)

        calibration_set = torch.tensor([])
        truth_labels = torch.tensor([])

        with pytest.raises(ValueError, match="calibration_set must not be empty"):
            calibrator.fit(calibration_set, truth_labels)

    def test_calibration_between_0_and_1(self) -> None:
        """Tests if the calibrator produces calibrated probabilities between 0 and 1.

        It checks that the calibrated probabilities produced by the calibrator after fitting are all within the range
        [0, 1].

        Raises:
            AssertionError: If any of the calibrated probabilities are outside the range [0, 1].
        """
        calibrator = HistogramBinningTorch(n_bins=5)

        calibration_set = torch.tensor([0.1, 0.4, 0.6, 0.8, 0.2, 0.9, 0.3, 0.5])
        truth_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])

        calibrator.fit(calibration_set, truth_labels)

        test_predictions = torch.tensor([0.15, 0.45, 0.65, 0.85])
        calibrated_probs = calibrator.predict(test_predictions)

        assert torch.all((calibrated_probs >= 0) & (calibrated_probs <= 1)), "Calibrated probabilities are not between "
        "0 and 1."

    def test_calibration_default_behaviour(self) -> None:
        """Tests the default behaviour of the histogram binning calibrator.

        It checks that the histogram binning calibrator produces expected calibrated probabilities after fitting
        on a simple calibration set and truth labels.

        Raises:
            AssertionError: If the calibrated probabilities do not match the expected values.
        """
        calibrator = HistogramBinningTorch(n_bins=4)

        calibration_set = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        truth_labels = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1])

        calibrator.fit(calibration_set, truth_labels)

        test_predictions = torch.tensor([0.15, 0.35, 0.55, 0.75])
        calibrated_probs = calibrator.predict(test_predictions)

        expected_probs = torch.tensor([0.0, 1.0, 0.5, 1.0])
        assert torch.allclose(calibrated_probs, expected_probs), "Calibrated probabilities do not match expected values"
