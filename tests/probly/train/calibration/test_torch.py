from __future__ import annotations

import pytest

from probly.train.calibration.torch import ExpectedCalibrationError, FocalLoss, LabelRelaxationLoss, LabelSmoothingLoss
from tests.probly.torch_utils import validate_loss

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402


def test_focal_loss(sample_outputs: tuple[Tensor, Tensor]) -> None:
    outputs, targets = sample_outputs
    criterion = FocalLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)
    # TODO(pwhofman): Add tests for different values of alpha and gamma
    # https://github.com/pwhofman/probly/issues/92


def test_expected_calibration_error(
    sample_outputs: tuple[Tensor, Tensor],
) -> None:
    outputs, targets = sample_outputs
    outputs = torch.softmax(outputs, dim=1)
    criterion = ExpectedCalibrationError()
    loss = criterion(outputs, targets)
    validate_loss(loss)

    criterion = ExpectedCalibrationError(num_bins=1)
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_expected_calibration_error_confidence_one() -> None:
    # Regression: torch.bucketize(1.0, linspace(0, 1, n_bins+1), right=True) - 1 == n_bins,
    # which would silently drop conf=1.0 samples from the loop and bias ECE downward.
    # All samples wrong with max-prob 1.0 -> perfect miscalibration -> ECE = 1.0.
    num_classes = 3
    probs = torch.zeros(4, num_classes)
    probs[:, 0] = 1.0
    targets = torch.ones(4, dtype=torch.long)  # never class 0 -> all wrong
    loss = ExpectedCalibrationError(num_bins=10)(probs, targets)
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_label_relaxation_loss(
    sample_outputs: tuple[Tensor, Tensor],
) -> None:
    outputs, targets = sample_outputs
    criterion = LabelRelaxationLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)

    criterion = LabelRelaxationLoss(alpha=1.0)
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_label_smoothing_loss(
    sample_outputs: tuple[Tensor, Tensor],
) -> None:
    outputs, targets = sample_outputs
    criterion = LabelSmoothingLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)

    expected = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, targets)
    torch.testing.assert_close(loss, expected)


def test_label_smoothing_loss_without_smoothing(
    sample_outputs: tuple[Tensor, Tensor],
) -> None:
    outputs, targets = sample_outputs
    loss = LabelSmoothingLoss(epsilon=0.0)(outputs, targets)
    expected = torch.nn.CrossEntropyLoss()(outputs, targets)
    torch.testing.assert_close(loss, expected)
