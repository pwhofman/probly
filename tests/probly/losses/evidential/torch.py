from __future__ import annotations

import pytest

from probly.layers.evidential.torch import BatchedRadialFlowDensity
from probly.losses.evidential.torch import (
    der_loss,
    dirichlet_entropy,
    dirichlet_prior_networks_loss,
    evidential_ce_loss,
    evidential_kl_divergence,
    evidential_log_loss,
    evidential_mse_loss,
    evidential_nignll_loss,
    evidential_regression_regularization,
    loss_ird,
    lp_fn,
    natpn_loss,
    postnet_loss,
    regularization_fn,
    rpn_distillation_loss,
    rpn_loss,
    rpn_ng_kl,
    rpn_prior,
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


def test_der_loss(
    torch_regression_model_1d: nn.Module,
    torch_regression_model_2d: nn.Module,
) -> None:
    inputs = torch.randn(4, 2)
    targets = torch.randn(4, 1)

    model: Predictor = evidential_regression(torch_regression_model_1d)
    outputs = model(inputs)
    criterion = der_loss
    loss = criterion(targets, *outputs)
    validate_loss(loss)

    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 1)

    model = evidential_regression(torch_regression_model_2d)
    outputs = model(inputs)
    loss = criterion(targets, *outputs)
    validate_loss(loss)


def test_rpn_distillation_loss() -> None:
    m = torch.randn(4, 1)
    l_precision = torch.rand(4, 1).abs() + 0.1
    kappa = torch.rand(4, 1).abs() + 0.1
    nu = torch.rand(4, 1).abs() + 3.0
    rpn_params = (m, l_precision, kappa, nu)

    mus = [torch.randn(4, 1) for _ in range(3)]
    variances = [torch.rand(4, 1).abs() + 0.01 for _ in range(3)]
    criterion = rpn_distillation_loss
    loss = criterion(rpn_params, mus, variances)
    validate_loss(loss)


def test_rpn_prior_returns_valid_parameters() -> None:
    device = torch.device("cpu")
    shape = (4, 1)

    mu0, kappa0, alpha0, beta0 = rpn_prior(shape, device)
    assert mu0.shape == shape
    assert kappa0.shape == shape
    assert alpha0.shape == shape
    assert beta0.shape == shape

    assert mu0.device == device
    assert kappa0.device == device
    assert alpha0.device == device
    assert beta0.device == device

    assert torch.isfinite(mu0).all()
    assert torch.isfinite(kappa0).all()
    assert torch.isfinite(alpha0).all()
    assert torch.isfinite(beta0).all()

    assert (kappa0 >= 0).all()
    assert (alpha0 > 1.0).all()
    assert (beta0 >= 0).all()


def test_rpn_ng_kl(
    torch_regression_model_1d: nn.Module,
    torch_regression_model_2d: nn.Module,
) -> None:
    inputs = torch.randn(4, 2)
    model: Predictor = evidential_regression(torch_regression_model_1d)
    mu, kappa, alpha, beta = model(inputs)

    mu0, kappa0, alpha0, beta0 = rpn_prior(mu.shape, mu.device)

    criterion = rpn_ng_kl
    loss = criterion(mu, kappa, alpha, beta, mu0, kappa0, alpha0, beta0)
    validate_loss(loss)

    inputs = torch.randn(4, 4)
    model = evidential_regression(torch_regression_model_2d)
    mu, kappa, alpha, beta = model(inputs)

    mu0, kappa0, alpha0, beta0 = rpn_prior(mu.shape, mu.device)

    criterion = rpn_ng_kl
    loss = criterion(mu, kappa, alpha, beta, mu0, kappa0, alpha0, beta0)

    validate_loss(loss)


def test_rpn_loss(
    torch_regression_model_1d: nn.Module,
    torch_regression_model_2d: nn.Module,
) -> None:
    model: Predictor = evidential_regression(torch_regression_model_1d)

    x_id = torch.randn(4, 2)
    y_id = torch.randn(4, 1)

    x_ood = torch.randn(4, 2) * 3.0 + 5.0

    criterion = rpn_loss
    loss = criterion(model, x_id, y_id, x_ood)

    validate_loss(loss)
    model = evidential_regression(torch_regression_model_2d)

    x_id = torch.randn(4, 4)
    y_id = torch.randn(4, 2)

    x_ood = torch.randn(4, 4) * 4.0 - 2.0
    criterion = rpn_loss
    loss = criterion(model, x_id, y_id, x_ood)
    validate_loss(loss)


def test_postnet_loss(
    sample_classification_data: tuple[Tensor, Tensor],
) -> None:
    inputs, targets = sample_classification_data
    batch_size = inputs.size(0)
    num_classes = 10  # Standard MNIST
    latent_dim = 16  # Standard latent dimension

    # Create flow density model
    flow = BatchedRadialFlowDensity(
        num_classes=num_classes,
        dim=latent_dim,
        flow_length=6,
    )

    # Create simulated network outputs (z): (batch_size, latent_dim)
    z = torch.randn(batch_size, latent_dim)

    # Create class counts for the batch
    class_counts = torch.zeros(num_classes)
    class_counts[targets] = 1  # Count how many times each class appears

    # Compute loss
    loss, alpha = postnet_loss(z, targets, flow, class_counts)

    # Validate the loss value
    validate_loss(loss)

    # Validate that alpha has correct shape
    expected_shape = (batch_size, num_classes)
    assert alpha.shape == expected_shape, f"Expected {expected_shape}, got {alpha.shape}"


def test_lp_fn(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    # Convert targets to one-hot encoding (lp_fn requires this)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1]).float()
    criterion = lp_fn
    loss = criterion(outputs, targets_onehot)
    validate_loss(loss)


def test_regularization_fn(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1]).float()
    criterion = regularization_fn
    loss = criterion(outputs, targets_onehot)
    validate_loss(loss)


def test_dirichlet_entropy(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, _ = sample_classification_data  # targets not needed for dirichlet_entropy
    outputs = evidential_classification_model(inputs)
    criterion = dirichlet_entropy
    loss = criterion(outputs)
    validate_loss(loss)


def test_loss_ird(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1]).float()
    criterion = loss_ird

    # Test 1: ID-only loss (without adversarial inputs)
    loss = criterion(outputs, targets_onehot)
    validate_loss(loss)

    # Test 2: ID + OOD (adversarial) loss
    # Generate adversarial/OOD inputs and get their alpha predictions
    adversarial_inputs = torch.randn_like(inputs)
    adversarial_alpha = evidential_classification_model(adversarial_inputs)
    loss_with_adv = criterion(outputs, targets_onehot, adversarial_alpha=adversarial_alpha)
    validate_loss(loss_with_adv)


def test_natpn_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = natpn_loss
    loss = criterion(outputs, targets)
    validate_loss(loss)


def test_dirichlet_prior_networks_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    criterion = dirichlet_prior_networks_loss

    # Test 1: ID-only loss (without OOD)
    loss = criterion(outputs, targets)
    validate_loss(loss)

    # Test 2: ID + OOD loss
    # Generate OOD inputs and get their alpha predictions
    ood_inputs = torch.randn_like(inputs)
    alpha_ood = evidential_classification_model(ood_inputs)
    loss_with_ood = criterion(outputs, targets, alpha_ood=alpha_ood)
    validate_loss(loss_with_ood)
