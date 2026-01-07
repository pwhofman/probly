from __future__ import annotations

import pytest

from probly.losses.evidential.torch import (
    evidential_ce_loss,
    evidential_kl_divergence,
    evidential_log_loss,
    evidential_mse_loss,
    evidential_nignll_loss,
    evidential_regression_regularization,
)
from probly.predictor import Predictor
from probly.transformation import evidential_regression
from tests.probly.torch_utils import validate_loss

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402


def test_evidential_log_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = evidential_log_loss
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_ce_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = evidential_ce_loss
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_mse_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = evidential_mse_loss
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_kl_divergence(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = evidential_kl_divergence
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_nig_nll_loss(
    torch_regression_model_1d: nn.Module,
    torch_regression_model_2d: nn.Module,
) -> None:
    inputs = torch.randn(2, 2)
    targets = torch.randn(2, 1)
    model: Predictor = evidential_regression(torch_regression_model_1d)
    outputs = model(inputs)
    criterion = evidential_nignll_loss
    loss = criterion(outputs, targets)
    validate_loss(loss)

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 2)
    model = evidential_regression(torch_regression_model_2d)
    outputs = model(inputs)
    criterion = evidential_nignll_loss
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_regression_regularization(
    torch_regression_model_1d: nn.Module,
    torch_regression_model_2d: nn.Module,
) -> None:
    inputs = torch.randn(2, 2)
    targets = torch.randn(2, 1)
    model: Predictor = evidential_regression(torch_regression_model_1d)
    outputs = model(inputs)
    criterion = evidential_regression_regularization
    loss = criterion(outputs, targets)
    validate_loss(loss)

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 2)
    model = evidential_regression(torch_regression_model_2d)
    outputs = model(inputs)
    criterion = evidential_regression_regularization
    loss = criterion(outputs, targets)
    validate_loss(loss)
