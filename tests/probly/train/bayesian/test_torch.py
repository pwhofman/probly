from __future__ import annotations

import pytest

from probly.method.bayesian import bayesian
from probly.train.bayesian.torch import elbo_loss
from probly.transformation.bayesian import collect_kl_divergence
from tests.probly.torch_utils import validate_loss

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402


def test_elbo_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    torch_conv_linear_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    model = bayesian(torch_conv_linear_model)
    outputs = model(inputs)

    validate_loss(elbo_loss(outputs, targets, collect_kl_divergence(model)))
    validate_loss(elbo_loss(outputs, targets, collect_kl_divergence(model), kl_penalty=0.0))
