"""Tests for ``PosteriorNetworkDecomposition`` and end-to-end PostNet smoke checks."""

from __future__ import annotations

from typing import cast

import pytest

from probly.method.posterior_network import PosteriorNetworkDecomposition, posterior_network

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.predictor import predict  # noqa: E402
from probly.quantification import (  # noqa: E402
    AleatoricUncertainty,
    EpistemicUncertainty,
    TotalUncertainty,
)
from probly.quantification.measure.distribution import (  # noqa: E402
    entropy_of_expected_predictive_distribution,
    max_probability_complement_of_expected,
    vacuity,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402
from probly.transformation.posterior_network import PosteriorNetworkPredictor  # noqa: E402

NUMERIC_BASES: tuple[None | float, ...] = (None, 2.0, 10.0)
NUM_CLASSES = 4
IN_FEATURES = 8
HIDDEN = 16
LATENT_DIM = 4
BATCH = 3


def _torch_dirichlet() -> TorchDirichletDistribution:
    alphas = torch.tensor(
        [
            [2.0, 3.0, 5.0],
            [1.0, 1.0, 1.0],
            [10.0, 10.0, 10.0],
            [100.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return TorchDirichletDistribution(alphas=alphas)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_torch_decomposition_components_match_measure_functions(base: None | float) -> None:
    distribution = _torch_dirichlet()

    decomposition = PosteriorNetworkDecomposition(distribution, base=base)

    assert torch.allclose(
        decomposition.total,
        max_probability_complement_of_expected(distribution),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(
        decomposition.aleatoric,
        entropy_of_expected_predictive_distribution(distribution, base=base),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(decomposition.epistemic, vacuity(distribution), rtol=1e-12, atol=1e-12)


def test_torch_decomposition_notion_access_and_types() -> None:
    decomposition = PosteriorNetworkDecomposition(_torch_dirichlet())

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
    decomposition = PosteriorNetworkDecomposition(_torch_dirichlet())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic


def test_torch_decomposition_canonical_notion_is_total() -> None:
    decomposition = PosteriorNetworkDecomposition(_torch_dirichlet())

    assert decomposition.get_canonical() is decomposition.total


def test_torch_decomposition_propagates_gradients() -> None:
    alphas = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    distribution = TorchDirichletDistribution(alphas=alphas)

    decomposition = PosteriorNetworkDecomposition(distribution)
    objective = decomposition.total + decomposition.aleatoric + decomposition.epistemic
    objective.backward()

    grad = alphas.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


class _TinyEncoder(nn.Module):
    """Two-layer MLP used as a stand-in encoder for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model() -> PosteriorNetworkPredictor:
    torch.manual_seed(0)
    encoder = _TinyEncoder()
    return cast(
        "PosteriorNetworkPredictor",
        posterior_network(
            encoder,
            latent_dim=LATENT_DIM,
            num_classes=NUM_CLASSES,
            class_counts=[100] * NUM_CLASSES,
            num_flows=2,
        ),
    )


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(1)
    # BatchNorm1d in PostNet requires batch size > 1
    return torch.randn(max(BATCH, 2), IN_FEATURES)


def test_postnet_decomposition_on_model_output(model: PosteriorNetworkPredictor, inputs: torch.Tensor) -> None:
    """End-to-end: a PostNet predictor's Dirichlet output feeds the decomposition cleanly."""
    model.eval()  # avoid BatchNorm running-stats mutation
    dirichlet = predict(model, inputs)
    assert isinstance(dirichlet, TorchDirichletDistribution)

    decomposition = PosteriorNetworkDecomposition(dirichlet)

    n = inputs.shape[0]
    assert decomposition.total.shape == (n,)
    assert decomposition.aleatoric.shape == (n,)
    assert decomposition.epistemic.shape == (n,)
    assert torch.all(decomposition.total >= 0.0)
    assert torch.all(decomposition.total < 1.0)
    assert torch.all(decomposition.epistemic > 0.0)
    assert torch.all(decomposition.epistemic <= 1.0)
