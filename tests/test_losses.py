"""Tests for the losses module."""

from __future__ import annotations

import pytest
import torch

from probly.losses import (
    ELBOLoss,
    EvidentialCELoss,
    EvidentialKLDivergence,
    EvidentialLogLoss,
    EvidentialMSELoss,
    EvidentialNIGNLLLoss,
    EvidentialRegressionRegularization,
    ExpectedCalibrationError,
    FocalLoss,
    LabelRelaxationLoss,
)
from probly.representation import Bayesian, classification, regression


@pytest.fixture
def sample_classification_data() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randn(2, 3, 5, 5)
    targets = torch.randint(0, 2, (2,))
    return inputs, targets


@pytest.fixture
def sample_outputs(
    conv_linear_model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = conv_linear_model(torch.randn(2, 3, 5, 5))
    targets = torch.randint(0, 2, (2,))
    return outputs, targets


@pytest.fixture
def evidential_classification_model(
    conv_linear_model: torch.nn.Module,
) -> classification.Evidential:
    model: classification.Evidential = classification.Evidential(conv_linear_model)
    return model


def validate_loss(loss: torch.Tensor) -> None:
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0


def test_evidential_log_loss(
    sample_classification_data: tuple[torch.Tensor, torch.Tensor],
    evidential_classification_model: torch.nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = EvidentialLogLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_ce_loss(
    sample_classification_data: tuple[torch.Tensor, torch.Tensor],
    evidential_classification_model: torch.nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = EvidentialCELoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_mse_loss(
    sample_classification_data: tuple[torch.Tensor, torch.Tensor],
    evidential_classification_model: torch.nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = EvidentialMSELoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_kl_divergence(
    sample_classification_data: tuple[torch.Tensor, torch.Tensor],
    evidential_classification_model: torch.nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = EvidentialKLDivergence()
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_nig_nll_loss(
    regression_model_1d: torch.nn.Module,
    regression_model_2d: torch.nn.Module,
) -> None:
    inputs = torch.randn(2, 2)
    targets = torch.randn(2, 1)
    model: regression.Evidential = regression.Evidential(regression_model_1d)
    outputs = model(inputs)
    criterion = EvidentialNIGNLLLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 2)
    model = regression.Evidential(regression_model_2d)
    outputs = model(inputs)
    criterion = EvidentialNIGNLLLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_evidential_regression_regularization(
    regression_model_1d: torch.nn.Module,
    regression_model_2d: torch.nn.Module,
) -> None:
    inputs = torch.randn(2, 2)
    targets = torch.randn(2, 1)
    model: regression.Evidential = regression.Evidential(regression_model_1d)
    outputs = model(inputs)
    criterion = EvidentialRegressionRegularization()
    loss = criterion(outputs, targets)
    validate_loss(loss)

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 2)
    model = regression.Evidential(regression_model_2d)
    outputs = model(inputs)
    criterion = EvidentialRegressionRegularization()
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_focal_loss(sample_outputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    outputs, targets = sample_outputs
    criterion = FocalLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)
    # TODO(pwhofman): Add tests for different values of alpha and gamma
    # https://github.com/pwhofman/probly/issues/92


def test_elbo_loss(
    sample_classification_data: tuple[torch.Tensor, torch.Tensor],
    conv_linear_model: torch.nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    model: Bayesian = Bayesian(conv_linear_model)
    outputs = model(inputs)

    criterion = ELBOLoss()
    loss = criterion(outputs, targets, model.kl_divergence)
    validate_loss(loss)

    criterion = ELBOLoss(0.0)
    loss = criterion(outputs, targets, model.kl_divergence)
    validate_loss(loss)


def test_expected_calibration_error(
    sample_outputs: tuple[torch.Tensor, torch.Tensor],
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
    sample_outputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outputs, targets = sample_outputs
    criterion = LabelRelaxationLoss()
    loss = criterion(outputs, targets)
    validate_loss(loss)

    criterion = LabelRelaxationLoss(alpha=1.0)
    loss = criterion(outputs, targets)
    validate_loss(loss)
