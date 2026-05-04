"""Smoke tests for the torch Natural Posterior Network."""

from __future__ import annotations

from typing import cast

import pytest

from probly.method.natural_posterior_network import NaturalPosteriorDecomposition, natural_posterior_network

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.train.evidential.torch import postnet_loss  # noqa: E402
from probly.transformation.natural_posterior_network.torch import TorchNaturalPosteriorNetwork  # noqa: E402

NUM_CLASSES = 4
IN_FEATURES = 8
HIDDEN = 16
LATENT_DIM = 6
BATCH = 3


class _TinyEncoder(nn.Module):
    """Two-layer MLP used as a stand-in encoder for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model() -> TorchNaturalPosteriorNetwork:
    torch.manual_seed(0)
    encoder = _TinyEncoder()
    return cast(
        "TorchNaturalPosteriorNetwork",
        natural_posterior_network(
            encoder,
            latent_dim=LATENT_DIM,
            num_classes=NUM_CLASSES,
            num_flows=2,
            predictor_type="probabilistic_classifier",
        ),
    )


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, IN_FEATURES)


def test_forward_shape_and_positivity(model: TorchNaturalPosteriorNetwork, inputs: torch.Tensor) -> None:
    """Forward returns a positive ``[B, K]`` Dirichlet alpha tensor with no NaN/Inf."""
    alpha = model(inputs)
    assert alpha.shape == (BATCH, NUM_CLASSES)
    assert torch.isfinite(alpha).all()
    assert (alpha > 0).all()


def test_alpha_floor_at_prior(model: TorchNaturalPosteriorNetwork, inputs: torch.Tensor) -> None:
    """Each ``alpha_k`` must be at least ``alpha_prior_k`` since ``n(x) * chi >= 0``."""
    alpha = model(inputs)
    assert (alpha >= model.alpha_prior - 1e-5).all()


def test_postnet_loss_backward(model: TorchNaturalPosteriorNetwork, inputs: torch.Tensor) -> None:
    """Backward through ``postnet_loss`` produces a finite gradient on the classifier."""
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    alpha = model(inputs)
    loss = postnet_loss(alpha, targets, entropy_weight=1e-5, reduction="mean")
    loss.backward()
    grad = model.classifier.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_certainty_budget_changes_evidence(inputs: torch.Tensor) -> None:
    """Different budgets produce different ``alpha`` for the same encoder weights."""
    torch.manual_seed(0)
    encoder_a = _TinyEncoder()
    torch.manual_seed(0)
    encoder_b = _TinyEncoder()

    model_constant = cast(
        "TorchNaturalPosteriorNetwork",
        natural_posterior_network(
            encoder_a,
            latent_dim=LATENT_DIM,
            num_classes=NUM_CLASSES,
            num_flows=2,
            certainty_budget="constant",
            predictor_type="probabilistic_classifier",
        ),
    )
    model_normal = cast(
        "TorchNaturalPosteriorNetwork",
        natural_posterior_network(
            encoder_b,
            latent_dim=LATENT_DIM,
            num_classes=NUM_CLASSES,
            num_flows=2,
            certainty_budget="normal",
            predictor_type="probabilistic_classifier",
        ),
    )

    alpha_constant = model_constant(inputs)
    alpha_normal = model_normal(inputs)
    # The "normal" budget adds a positive log_scale at H=6, so its alphas must
    # differ from (and have larger evidence above the prior than) "constant".
    assert not torch.allclose(alpha_constant, alpha_normal)
    evidence_constant = (alpha_constant - 1.0).sum(-1)
    evidence_normal = (alpha_normal - 1.0).sum(-1)
    assert (evidence_normal > evidence_constant).all()


# Tests for NaturalPosteriorDecomposition on Torch Dirichlet representations.

from probly.quantification import (  # noqa: E402
    AleatoricUncertainty,
    EpistemicUncertainty,
    TotalUncertainty,
)
from probly.quantification.measure.distribution import (  # noqa: E402
    entropy,
    entropy_of_expected_predictive_distribution,
    vacuity,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402

NUMERIC_BASES: tuple[None | float, ...] = (None, 2.0, 10.0)


def _torch_dirichlet() -> TorchDirichletDistribution:
    alphas = torch.tensor(
        [
            [2.0, 3.0, 5.0],
            [1.0, 1.0, 1.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=torch.float64,
    )
    return TorchDirichletDistribution(alphas=alphas)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_torch_decomposition_components_match_measure_functions(base: None | float) -> None:
    distribution = _torch_dirichlet()

    decomposition = NaturalPosteriorDecomposition(distribution, base=base)

    assert torch.allclose(decomposition.total, entropy(distribution, base=base), rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        decomposition.aleatoric,
        entropy_of_expected_predictive_distribution(distribution, base=base),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(decomposition.epistemic, vacuity(distribution), rtol=1e-12, atol=1e-12)


def test_torch_decomposition_notion_access_and_types() -> None:
    decomposition = NaturalPosteriorDecomposition(_torch_dirichlet())

    assert isinstance(decomposition.total, torch.Tensor)
    assert isinstance(decomposition.aleatoric, torch.Tensor)
    assert isinstance(decomposition.epistemic, torch.Tensor)
    assert decomposition[TotalUncertainty] is decomposition.total
    assert decomposition[AleatoricUncertainty] is decomposition.aleatoric
    assert decomposition[EpistemicUncertainty] is decomposition.epistemic
    assert decomposition["tu"] is decomposition.total
    assert decomposition["au"] is decomposition.aleatoric
    assert decomposition["eu"] is decomposition.epistemic


def test_torch_decomposition_caches_components() -> None:
    decomposition = NaturalPosteriorDecomposition(_torch_dirichlet())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic


def test_torch_decomposition_canonical_notion_is_total() -> None:
    decomposition = NaturalPosteriorDecomposition(_torch_dirichlet())

    assert decomposition.get_canonical() is decomposition.total


def test_torch_decomposition_propagates_gradients() -> None:
    alphas = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    distribution = TorchDirichletDistribution(alphas=alphas)

    decomposition = NaturalPosteriorDecomposition(distribution)
    objective = decomposition.total + decomposition.aleatoric + decomposition.epistemic
    objective.backward()

    assert alphas.grad is not None
    assert torch.isfinite(alphas.grad).all()


def test_torch_decomposition_on_natpn_model_output(model: TorchNaturalPosteriorNetwork, inputs: torch.Tensor) -> None:
    """End-to-end: NatPN output Dirichlet feeds the NatPN decomposition."""
    alphas = model(inputs)
    distribution = TorchDirichletDistribution(alphas=alphas)

    decomposition = NaturalPosteriorDecomposition(distribution)

    assert decomposition.total.shape == (BATCH,)
    assert decomposition.aleatoric.shape == (BATCH,)
    assert decomposition.epistemic.shape == (BATCH,)
    assert torch.all(decomposition.epistemic > 0.0)
    assert torch.all(decomposition.epistemic <= 1.0)
