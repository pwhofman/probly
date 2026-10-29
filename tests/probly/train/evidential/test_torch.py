from __future__ import annotations

import pytest

from probly.layers.torch import RadialNormalizingFlowStack
from probly.method.evidential import evidential_regression
from probly.predictor import Predictor
from probly.train.evidential.torch import (
    der_loss,
    dirichlet_entropy,
    evidential_ce_loss,
    evidential_kl_divergence,
    evidential_log_loss,
    evidential_mse_loss,
    evidential_nignll_loss,
    evidential_regression_regularization,
    ird_loss,
    lp_fn,
    natpn_loss,
    postnet_loss,
    regularization_fn,
    rpn_distillation_loss,
    rpn_loss,
    rpn_ng_kl,
    rpn_prior,
)
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_postnet_loss(
    sample_classification_data: tuple[Tensor, Tensor],
) -> None:
    inputs, targets = sample_classification_data
    batch_size = inputs.size(0)
    num_classes = 10  # Standard MNIST
    latent_dim = 16  # Standard latent dimension

    # Create flow density model
    flow = RadialNormalizingFlowStack(
        num_classes=num_classes,
        dim=latent_dim,
        num_flows=6,
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


@pytest.mark.skip
def test_dirichlet_entropy(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, _ = sample_classification_data  # targets not needed for dirichlet_entropy
    outputs = evidential_classification_model(inputs)
    criterion = dirichlet_entropy
    loss = criterion(outputs)
    validate_loss(loss)


def test_ird_loss(
    sample_classification_data: tuple[Tensor, Tensor],
    evidential_classification_model: nn.Module,
) -> None:
    inputs, targets = sample_classification_data
    outputs = evidential_classification_model(inputs)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1]).float()
    criterion = ird_loss

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


# ---------------------------------------------------------------------------
# Helper-function tests merged from test_torch_helpers.py.
# ---------------------------------------------------------------------------


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    return torch


class TestMakeInDomainTargetAlpha:
    def test_uses_explicit_num_classes_not_batch_labels(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import make_in_domain_target_alpha  # noqa: PLC0415

        # Labels only span 0..1 but the dataset has 5 classes.
        y = torch.tensor([0, 1])
        alpha = make_in_domain_target_alpha(y, num_classes=5)
        assert alpha.shape == (2, 5)

    def test_target_is_concentration_plus_beta_on_true_class(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import make_in_domain_target_alpha  # noqa: PLC0415

        # Malinin and Gales (2019), eq. 7: beta + 1 on the true class, 1 elsewhere.
        y = torch.tensor([0, 2])
        alpha = make_in_domain_target_alpha(y, num_classes=3)
        torch.testing.assert_close(
            alpha,
            torch.tensor(
                [
                    [101.0, 1.0, 1.0],
                    [1.0, 1.0, 101.0],
                ]
            ),
        )

    def test_custom_concentrations(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import make_in_domain_target_alpha  # noqa: PLC0415

        y = torch.tensor([1])
        alpha = make_in_domain_target_alpha(y, num_classes=2, target_concentration=1e3, concentration=0.5)
        torch.testing.assert_close(alpha, torch.tensor([[0.5, 1000.5]]))


class TestMakeOodTargetAlpha:
    def test_flat_dirichlet_of_ones(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import make_ood_target_alpha  # noqa: PLC0415

        out = make_ood_target_alpha(batch_size=4, num_classes=7)
        assert out.shape == (4, 7)
        torch.testing.assert_close(out, torch.ones(4, 7))

    def test_custom_concentration(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import make_ood_target_alpha  # noqa: PLC0415

        out = make_ood_target_alpha(batch_size=2, num_classes=5, concentration=0.5)
        torch.testing.assert_close(out, torch.full((2, 5), 0.5))


class TestPredictiveProbs:
    def test_normalises_alpha(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import predictive_probs  # noqa: PLC0415

        alpha = torch.tensor([[1.0, 2.0, 3.0], [10.0, 0.0, 0.0]])
        probs = predictive_probs(alpha)
        torch.testing.assert_close(probs, alpha / alpha.sum(dim=-1, keepdim=True))
        torch.testing.assert_close(probs.sum(dim=-1), torch.tensor([1.0, 1.0]))


class TestKLDirichlet:
    def test_zero_divergence_for_identical(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import kl_dirichlet  # noqa: PLC0415

        a = torch.tensor([[2.0, 3.0]])
        kl = kl_dirichlet(a, a)
        # KL(p || p) = 0
        torch.testing.assert_close(kl.squeeze(), torch.tensor(0.0), atol=1e-5, rtol=1e-5)

    def test_positive_divergence_for_different(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import kl_dirichlet  # noqa: PLC0415

        prior = torch.tensor([[2.0, 3.0]])
        posterior = torch.tensor([[5.0, 5.0]])
        kl = kl_dirichlet(prior, posterior)
        # KL(p, q) >= 0 and != 0 when p != q.
        assert kl.item() > 0


class TestEvidentialLogLoss:
    def test_smaller_for_correct_class(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import evidential_log_loss  # noqa: PLC0415

        # alphas with high concentration on target -> small log loss.
        alphas_sharp = torch.tensor([[10.0, 1.0, 1.0]])
        alphas_flat = torch.tensor([[1.0, 1.0, 1.0]])
        targets = torch.tensor([0])
        loss_sharp = evidential_log_loss(alphas_sharp, targets)
        loss_flat = evidential_log_loss(alphas_flat, targets)
        assert loss_sharp.item() < loss_flat.item()


class TestEvidentialCELoss:
    def test_smaller_for_correct_class(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import evidential_ce_loss  # noqa: PLC0415

        alphas_sharp = torch.tensor([[10.0, 1.0, 1.0]])
        alphas_flat = torch.tensor([[1.0, 1.0, 1.0]])
        targets = torch.tensor([0])
        loss_sharp = evidential_ce_loss(alphas_sharp, targets)
        loss_flat = evidential_ce_loss(alphas_flat, targets)
        assert loss_sharp.item() < loss_flat.item()


class TestEvidentialMSELoss:
    def test_returns_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import evidential_mse_loss  # noqa: PLC0415

        alphas = torch.tensor([[10.0, 1.0, 1.0]])
        targets = torch.tensor([0])
        loss = evidential_mse_loss(alphas, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestEvidentialKLDivergence:
    def test_returns_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import evidential_kl_divergence  # noqa: PLC0415

        alphas = torch.tensor([[2.0, 1.0, 1.0]])
        targets = torch.tensor([0])
        loss = evidential_kl_divergence(alphas, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestLpFn:
    def test_finite(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lp_fn  # noqa: PLC0415

        alphas = torch.tensor([[2.0, 3.0, 1.0]])
        y_onehot = torch.tensor([[1.0, 0.0, 0.0]])  # class 0 one-hot
        loss = lp_fn(alphas, y_onehot, p=2.0)
        assert torch.isfinite(loss)

    def test_smaller_for_correct_class(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lp_fn  # noqa: PLC0415

        sharp = torch.tensor([[10.0, 1.0, 1.0]])
        flat = torch.tensor([[1.0, 1.0, 1.0]])
        y_onehot = torch.tensor([[1.0, 0.0, 0.0]])
        loss_sharp = lp_fn(sharp, y_onehot, p=2.0)
        loss_flat = lp_fn(flat, y_onehot, p=2.0)
        assert loss_sharp.item() < loss_flat.item()

    def test_non_positive_alpha_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lp_fn  # noqa: PLC0415

        alphas = torch.tensor([[0.0, 1.0, 1.0]])
        y_onehot = torch.tensor([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match=r"alpha values must be > 0"):
            lp_fn(alphas, y_onehot)

    def test_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lp_fn  # noqa: PLC0415

        with pytest.raises(ValueError, match="shape mismatch"):
            lp_fn(torch.tensor([[1.0, 1.0]]), torch.tensor([0]))


class TestRegularizationFn:
    def test_finite(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import regularization_fn  # noqa: PLC0415

        alphas = torch.tensor([[2.0, 3.0, 1.0]])
        y_onehot = torch.tensor([[1.0, 0.0, 0.0]])
        reg = regularization_fn(alphas, y_onehot)
        assert torch.isfinite(reg)


class TestDirichletEntropy:
    def test_higher_entropy_for_uniform(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import dirichlet_entropy  # noqa: PLC0415

        sharp = torch.tensor([[10.0, 1.0, 1.0]])
        flat = torch.tensor([[1.0, 1.0, 1.0]])
        h_sharp = dirichlet_entropy(sharp).sum()
        h_flat = dirichlet_entropy(flat).sum()
        # Uniform Dirichlet has higher entropy than peaked Dirichlet.
        assert h_flat.item() > h_sharp.item()


class TestEvidentialNigNllLoss:
    def test_finite_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import evidential_nignll_loss  # noqa: PLC0415

        # Inputs is a dict with gamma, nu, alpha, beta keys.
        gamma = torch.tensor([[1.0]])
        nu = torch.tensor([[1.0]])
        alpha = torch.tensor([[2.0]])
        beta = torch.tensor([[1.0]])
        inputs = {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}
        targets = torch.tensor([[1.0]])
        loss = evidential_nignll_loss(inputs, targets)
        assert torch.isfinite(loss).all()


class TestEvidentialRegressionRegularization:
    def test_finite_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import evidential_regression_regularization  # noqa: PLC0415

        gamma = torch.tensor([[1.0]])
        nu = torch.tensor([[1.0]])
        alpha = torch.tensor([[2.0]])
        beta = torch.tensor([[1.0]])
        inputs = {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}
        targets = torch.tensor([[1.0]])
        reg = evidential_regression_regularization(inputs, targets)
        assert torch.isfinite(reg).all()


# ---------------------------------------------------------------------------
# Coverage for the dispatch entrypoint, training-path utilities, and helpers
# inside the loss functions.
# ---------------------------------------------------------------------------


class TestComputeLossDispatch:
    """The compute_loss switchdispatch fallback raises on unknown modes."""

    def test_unknown_mode_raises(self) -> None:
        from probly.train.evidential.torch import compute_loss  # noqa: PLC0415

        with pytest.raises(ValueError, match="Enter a valid mode"):
            compute_loss("not_a_known_mode")


class TestPostNetLoss:
    """postnet_loss supports both 'mean' and 'sum' reductions."""

    def test_default_sum_returns_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import postnet_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 3.0, 1.0]])
        y = torch.tensor([0, 1])
        loss = postnet_loss(alpha, y)  # default reduction='sum'
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    def test_mean_reduction(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import postnet_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 3.0, 1.0]])
        y = torch.tensor([0, 1])
        loss_mean = postnet_loss(alpha, y, reduction="mean")
        loss_sum = postnet_loss(alpha, y, reduction="sum")
        # mean reduction divides the same per-sample loss by batch size.
        torch.testing.assert_close(loss_mean, loss_sum / 2.0)


class TestMixtureUceLoss:
    """mixture_uce_loss supports 'mean', 'sum', and 'none' reductions and
    raises on unsupported reductions.
    """  # noqa: D205

    def test_sum_returns_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])  # (N=2, C=3)
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])  # (B=2, N=2)
        y = torch.tensor([0, 1])
        loss = mixture_uce_loss(alpha, mixture_weights, y, reduction="sum")
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    def test_mean_returns_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        y = torch.tensor([0, 1])
        loss = mixture_uce_loss(alpha, mixture_weights, y, reduction="mean")
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    def test_none_returns_per_sample(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        y = torch.tensor([0, 1])
        loss = mixture_uce_loss(alpha, mixture_weights, y, reduction="none")
        # No reduction -> one loss per batch element.
        assert loss.shape == (2,)
        assert torch.isfinite(loss).all()

    def test_unsupported_reduction_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])
        mixture_weights = torch.tensor([[1.0]])
        y = torch.tensor([0])
        with pytest.raises(ValueError, match="Unsupported reduction"):
            mixture_uce_loss(alpha, mixture_weights, y, reduction="bogus")


class TestLopGpnLoss:
    """lop_gpn_loss combines mixture_uce_loss with optional entropy regularization."""

    def test_no_regularization_returns_uce(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lop_gpn_loss, mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        y = torch.tensor([0, 1])
        # Without entropy_regularization, lop_gpn_loss equals mixture_uce_loss.
        loss = lop_gpn_loss(alpha, mixture_weights, y, reduction="sum")
        baseline = mixture_uce_loss(alpha, mixture_weights, y, reduction="sum")
        torch.testing.assert_close(loss, baseline)

    def test_entropy_zero_weight_returns_uce(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lop_gpn_loss, mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        y = torch.tensor([0, 1])
        entropy = torch.tensor([0.5, 0.7])
        # entropy_weight=0.0 short-circuits even when a regulariser is given.
        loss = lop_gpn_loss(
            alpha,
            mixture_weights,
            y,
            entropy_regularization=entropy,
            entropy_weight=0.0,
            reduction="sum",
        )
        baseline = mixture_uce_loss(alpha, mixture_weights, y, reduction="sum")
        torch.testing.assert_close(loss, baseline)

    def test_with_entropy_sum_reduction(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lop_gpn_loss, mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        y = torch.tensor([0, 1])
        entropy = torch.tensor([0.5, 0.7])
        loss = lop_gpn_loss(
            alpha,
            mixture_weights,
            y,
            entropy_regularization=entropy,
            entropy_weight=0.1,
            reduction="sum",
        )
        baseline = mixture_uce_loss(alpha, mixture_weights, y, reduction="sum")
        # sum reduction: subtracts entropy_weight * entropy.sum().
        torch.testing.assert_close(loss, baseline - 0.1 * entropy.sum())

    def test_with_entropy_mean_reduction(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lop_gpn_loss, mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        y = torch.tensor([0, 1])
        entropy = torch.tensor([0.5, 0.7])
        loss = lop_gpn_loss(
            alpha,
            mixture_weights,
            y,
            entropy_regularization=entropy,
            entropy_weight=0.1,
            reduction="mean",
        )
        baseline = mixture_uce_loss(alpha, mixture_weights, y, reduction="mean")
        torch.testing.assert_close(loss, baseline - 0.1 * entropy.mean())

    def test_with_entropy_none_reduction(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lop_gpn_loss, mixture_uce_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        mixture_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        y = torch.tensor([0, 1])
        entropy = torch.tensor([0.5, 0.7])
        loss = lop_gpn_loss(
            alpha,
            mixture_weights,
            y,
            entropy_regularization=entropy,
            entropy_weight=0.1,
            reduction="none",
        )
        baseline = mixture_uce_loss(alpha, mixture_weights, y, reduction="none")
        # Per-sample loss shape preserved.
        torch.testing.assert_close(loss, baseline - 0.1 * entropy)
        assert loss.shape == (2,)

    def test_unsupported_reduction_with_entropy_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lop_gpn_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])
        mixture_weights = torch.tensor([[1.0]])
        y = torch.tensor([0])
        # mixture_uce_loss(reduction='bogus') would raise first; force
        # reduction to a value that mixture_uce_loss accepts but lop_gpn_loss's
        # entropy block does not... no such value exists, so we test by
        # bypassing the early check: call with valid reduction and entropy
        # weight 0 is already covered.  Instead simulate the explicit check by
        # making mixture_uce_loss return early -- but reduction is shared. So
        # we cover the fallthrough error path by passing an unsupported value
        # while supplying an entropy regulariser. The error originates from
        # mixture_uce_loss, which validates reduction first.
        with pytest.raises(ValueError, match="Unsupported reduction"):
            lop_gpn_loss(
                alpha,
                mixture_weights,
                y,
                entropy_regularization=torch.tensor([0.5]),
                entropy_weight=0.1,
                reduction="bogus",
            )


class TestRegularizationFnValidation:
    """regularization_fn raises a clear error when shapes mismatch."""

    def test_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import regularization_fn  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])  # (1, 3)
        y = torch.tensor([[1.0, 0.0]])  # (1, 2) -> mismatch
        with pytest.raises(ValueError, match="shape mismatch"):
            regularization_fn(alpha, y)


class TestDirichletEntropyValidation:
    """dirichlet_entropy validates positivity of alpha."""

    def test_non_positive_alpha_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import dirichlet_entropy  # noqa: PLC0415

        with pytest.raises(ValueError, match=r"alpha values must be > 0"):
            dirichlet_entropy(torch.tensor([[0.0, 1.0, 1.0]]))


class TestIRDValidation:
    """ird_loss checks input shapes and adversarial alpha shapes."""

    def test_alpha_must_be_2d(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import ird_loss  # noqa: PLC0415

        # alpha 1D -> error.
        alpha = torch.tensor([2.0, 1.0, 1.0])
        y = torch.tensor([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match=r"must be 2D"):
            ird_loss(alpha, y)

    def test_y_must_be_2d(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import ird_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])
        # y 1D -> error.
        y = torch.tensor([0])
        with pytest.raises(ValueError, match=r"must be 2D"):
            ird_loss(alpha, y)

    def test_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import ird_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])
        y = torch.tensor([[1.0, 0.0]])  # mismatched class dim
        with pytest.raises(ValueError, match="shape mismatch"):
            ird_loss(alpha, y)

    def test_non_positive_alpha_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import ird_loss  # noqa: PLC0415

        alpha = torch.tensor([[0.0, 1.0, 1.0]])
        y = torch.tensor([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match=r"alpha values must be > 0"):
            ird_loss(alpha, y)

    def test_adversarial_alpha_must_be_2d(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import ird_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])
        y = torch.tensor([[1.0, 0.0, 0.0]])
        adversarial = torch.tensor([2.0, 1.0, 1.0])  # 1D
        with pytest.raises(ValueError, match=r"adversarial_alpha must be 2D"):
            ird_loss(alpha, y, adversarial_alpha=adversarial)

    def test_adversarial_alpha_class_count_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import ird_loss  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])
        y = torch.tensor([[1.0, 0.0, 0.0]])
        adversarial = torch.tensor([[2.0, 1.0]])  # different class count
        with pytest.raises(ValueError, match="same number of classes"):
            ird_loss(alpha, y, adversarial_alpha=adversarial)


class TestLpFnDimensionMismatchVariant:
    """lp_fn 2D shape mismatch path with both 2D inputs but different shapes."""

    def test_2d_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import lp_fn  # noqa: PLC0415

        alpha = torch.tensor([[2.0, 1.0, 1.0]])  # (1,3)
        y = torch.tensor([[1.0, 0.0]])  # (1,2)
        with pytest.raises(ValueError, match="shape mismatch"):
            lp_fn(alpha, y)


class TestDerLoss:
    """der_loss returns a finite scalar Deep Evidential Regression loss."""

    def test_returns_finite_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import der_loss  # noqa: PLC0415

        y = torch.tensor([1.0, 2.0])
        mu = torch.tensor([1.1, 1.9])
        kappa = torch.tensor([1.0, 1.5])
        alpha = torch.tensor([2.0, 2.5])
        beta = torch.tensor([1.0, 1.0])
        loss = der_loss(y, mu, kappa, alpha, beta, lam=0.1)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()


class TestRpnLoss:
    """rpn_loss runs the full ID + OOD pipeline through a tiny NIG model."""

    def test_rpn_loss_finite(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import rpn_loss  # noqa: PLC0415

        # Lightweight stand-in for a NIG regressor: returns (mu, kappa, alpha, beta).
        class _TinyNIG(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(2, 4)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                out = self.lin(x)
                mu = out[:, 0]
                # softplus to keep positive parameters strictly positive.
                kappa = torch.nn.functional.softplus(out[:, 1]) + 1e-3
                alpha = torch.nn.functional.softplus(out[:, 2]) + 1.0
                beta = torch.nn.functional.softplus(out[:, 3]) + 1e-3
                return mu, kappa, alpha, beta

        model = _TinyNIG()
        x_id = torch.randn(4, 2)
        y_id = torch.randn(4)
        x_ood = torch.randn(4, 2)
        loss = rpn_loss(model, x_id, y_id, x_ood)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()


class TestRpnNgKl:
    """rpn_ng_kl computes a finite KL between two Normal-Gamma distributions."""

    def test_returns_finite_scalar(self) -> None:
        torch = _torch_modules()
        from probly.train.evidential.torch import rpn_ng_kl, rpn_prior  # noqa: PLC0415

        mu = torch.tensor([0.5, -0.5])
        kappa = torch.tensor([1.0, 2.0])
        alpha = torch.tensor([2.0, 2.5])
        beta = torch.tensor([1.0, 0.5])
        mu0, kappa0, alpha0, beta0 = rpn_prior(mu.shape, mu.device)
        loss = rpn_ng_kl(mu, kappa, alpha, beta, mu0, kappa0, alpha0, beta0)
        assert loss.shape == ()
        assert torch.isfinite(loss).all()


class _ExpDirNet(nn.Module):
    """Linear logits followed by exp, the Prior Network alpha parameterization."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(self.lin(x))


class TestPnLoss:
    """pn_loss follows Malinin and Gales (2019) and the authors' reference implementation."""

    def _setup(self, num_classes: int = 5, seed: int = 0):
        torch.manual_seed(seed)
        model = _ExpDirNet(4, num_classes)
        x_in = torch.randn(6, 4)
        y_in = torch.randint(0, num_classes, (6,))
        x_ood = torch.randn(6, 4)
        return model, x_in, y_in, x_ood

    def test_class_count_comes_from_model_output(self) -> None:
        from probly.train.evidential.torch import pn_loss  # noqa: PLC0415

        # 5-class model, batch labels never reach class 4: must not raise a shape error.
        model = _ExpDirNet(4, 5)
        loss = pn_loss(model, torch.randn(4, 4), torch.tensor([0, 1, 2, 3]), torch.randn(4, 4))
        assert loss.shape == ()
        assert torch.isfinite(loss).all()

    def test_reverse_kl_is_default(self) -> None:
        from torch.distributions import Dirichlet, kl_divergence  # noqa: PLC0415

        from probly.train.evidential.torch import (  # noqa: PLC0415
            make_in_domain_target_alpha,
            make_ood_target_alpha,
            pn_loss,
        )

        model, x_in, y_in, x_ood = self._setup()
        with torch.no_grad():
            alpha_in, alpha_ood = model(x_in), model(x_ood)
            target_in = make_in_domain_target_alpha(y_in, num_classes=5)
            target_ood = make_ood_target_alpha(x_ood.size(0), num_classes=5)
            # Eq. 10 of the 2019 paper: KL(model || target), normalised by the target concentration.
            kl_in = kl_divergence(Dirichlet(alpha_in), Dirichlet(target_in)).mean()
            kl_ood = kl_divergence(Dirichlet(alpha_ood), Dirichlet(target_ood)).mean()
            expected = (kl_in + kl_ood) / 1e2
            loss = pn_loss(model, x_in, y_in, x_ood)
        torch.testing.assert_close(loss, expected)

    def test_forward_kl_option(self) -> None:
        from torch.distributions import Dirichlet, kl_divergence  # noqa: PLC0415

        from probly.train.evidential.torch import (  # noqa: PLC0415
            make_in_domain_target_alpha,
            make_ood_target_alpha,
            pn_loss,
        )

        model, x_in, y_in, x_ood = self._setup()
        with torch.no_grad():
            alpha_in, alpha_ood = model(x_in), model(x_ood)
            target_in = make_in_domain_target_alpha(y_in, num_classes=5)
            target_ood = make_ood_target_alpha(x_ood.size(0), num_classes=5)
            # Eq. 12 of the 2018 paper: KL(target || model).
            kl_in = kl_divergence(Dirichlet(target_in), Dirichlet(alpha_in)).mean()
            kl_ood = kl_divergence(Dirichlet(target_ood), Dirichlet(alpha_ood)).mean()
            expected = (kl_in + kl_ood) / 1e2
            loss = pn_loss(model, x_in, y_in, x_ood, reverse=False)
        torch.testing.assert_close(loss, expected)

    def test_gamma_weights_ood_term(self) -> None:
        from torch.distributions import Dirichlet, kl_divergence  # noqa: PLC0415

        from probly.train.evidential.torch import make_ood_target_alpha, pn_loss  # noqa: PLC0415

        model, x_in, y_in, x_ood = self._setup()
        with torch.no_grad():
            alpha_ood = model(x_ood)
            target_ood = make_ood_target_alpha(x_ood.size(0), num_classes=5)
            kl_ood = kl_divergence(Dirichlet(alpha_ood), Dirichlet(target_ood)).mean()
            diff = pn_loss(model, x_in, y_in, x_ood, gamma=3.0) - pn_loss(model, x_in, y_in, x_ood, gamma=1.0)
        torch.testing.assert_close(diff, 2.0 * kl_ood / 1e2)

    def test_target_concentration_normalises_loss(self) -> None:
        from torch.distributions import Dirichlet, kl_divergence  # noqa: PLC0415

        from probly.train.evidential.torch import (  # noqa: PLC0415
            make_in_domain_target_alpha,
            make_ood_target_alpha,
            pn_loss,
        )

        model, x_in, y_in, x_ood = self._setup()
        with torch.no_grad():
            alpha_in, alpha_ood = model(x_in), model(x_ood)
            target_in = make_in_domain_target_alpha(y_in, num_classes=5, target_concentration=1e3)
            target_ood = make_ood_target_alpha(x_ood.size(0), num_classes=5)
            kl_in = kl_divergence(Dirichlet(alpha_in), Dirichlet(target_in)).mean()
            kl_ood = kl_divergence(Dirichlet(alpha_ood), Dirichlet(target_ood)).mean()
            expected = (kl_in + kl_ood) / 1e3
            loss = pn_loss(model, x_in, y_in, x_ood, target_concentration=1e3)
        torch.testing.assert_close(loss, expected)

    def test_backward_gives_finite_gradients(self) -> None:
        from probly.train.evidential.torch import pn_loss  # noqa: PLC0415

        model, x_in, y_in, x_ood = self._setup()
        loss = pn_loss(model, x_in, y_in, x_ood)
        loss.backward()
        grad = model.lin.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    def test_pn_loss_from_alphas_matches_pn_loss(self) -> None:
        from probly.train.evidential.torch import pn_loss, pn_loss_from_alphas  # noqa: PLC0415

        model, x_in, y_in, x_ood = self._setup()
        with torch.no_grad():
            expected = pn_loss(model, x_in, y_in, x_ood, gamma=3.0, target_concentration=1e3, reverse=False)
            loss = pn_loss_from_alphas(
                model(x_in), y_in, model(x_ood), gamma=3.0, target_concentration=1e3, reverse=False
            )
        torch.testing.assert_close(loss, expected)
