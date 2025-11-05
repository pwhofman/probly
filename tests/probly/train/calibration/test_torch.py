from __future__ import annotations

import pytest

from probly.train.calibration.torch import ExpectedCalibrationError, FocalLoss, LabelRelaxationLoss
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
