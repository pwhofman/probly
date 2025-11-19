from __future__ import annotations

import pytest

from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence
from probly.transformation import bayesian
from tests.probly.torch_utils import validate_loss

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402


def test_elbo_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    torch_conv_linear_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    model: nn.Module = bayesian(torch_conv_linear_model)
    outputs = model(inputs)

    criterion = ELBOLoss()
    loss = criterion(outputs, targets, collect_kl_divergence(model))
    validate_loss(loss)

    criterion = ELBOLoss(0.0)
    loss = criterion(outputs, targets, collect_kl_divergence(model))
    validate_loss(loss)
