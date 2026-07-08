from __future__ import annotations

import pytest

from probly.method.bayesian import bayesian
from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence
from tests.probly.torch_utils import validate_loss

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402


def test_collect_kl_divergence_gradients_reach_model(torch_conv_linear_model: nn.Module) -> None:
    model = bayesian(torch_conv_linear_model)

    kl = collect_kl_divergence(model)
    kl.backward()

    variational_params = {name: p for name, p in model.named_parameters() if name.endswith(("_mu", "_rho"))}
    assert variational_params
    for name, param in variational_params.items():
        assert param.grad is not None, f"kl.backward() left no gradient on {name}"


def test_elbo_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    torch_conv_linear_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    model = bayesian(torch_conv_linear_model)
    outputs = model(inputs)

    criterion = ELBOLoss()
    loss = criterion(outputs, targets, collect_kl_divergence(model))
    validate_loss(loss)

    criterion = ELBOLoss(0.0)
    loss = criterion(outputs, targets, collect_kl_divergence(model))
    validate_loss(loss)
