from __future__ import annotations

import pytest

from probly.train.calibration.torch import focal_loss, label_relaxation_loss
from tests.probly.torch_utils import validate_loss

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402


def test_focal_loss(sample_outputs: tuple[Tensor, Tensor]) -> None:
    outputs, targets = sample_outputs
    validate_loss(focal_loss(outputs, targets))
    # TODO(pwhofman): Add tests for different values of alpha and gamma
    # https://github.com/pwhofman/probly/issues/92


def test_label_relaxation_loss(
    sample_outputs: tuple[Tensor, Tensor],
) -> None:
    outputs, targets = sample_outputs
    validate_loss(label_relaxation_loss(outputs, targets))
    validate_loss(label_relaxation_loss(outputs, targets, alpha=1.0))
