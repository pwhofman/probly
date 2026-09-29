"""Tests for torch HetNets prediction and quantification."""

from __future__ import annotations

import math
from typing import Literal, cast

import pytest

from probly.method.het_net import HetNetRepresentation, het_net
from probly.predictor import predict
from probly.quantification import decompose, measure, quantify
from probly.quantification.decomposition.entropy import LabelNoiseEntropyDecomposition
from probly.representer import representer

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.layers.torch import HeteroscedasticLayer  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402


def _het_net_predictor(num_samples: int = 3) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 3),
    )
    return cast("nn.Module", het_net(model, num_factors=2, num_samples=num_samples, predictor_type="logit_classifier"))


def _het_net_representation() -> HetNetRepresentation:
    return representer(_het_net_predictor()).represent(torch.ones(2, 2))


def _noise_free_layer(predictor: nn.Module, mu: torch.Tensor) -> HeteroscedasticLayer:
    """Make the heteroscedastic layer of ``predictor`` return ``mu`` with negligible noise."""
    layer = next(m for m in predictor.modules() if isinstance(m, HeteroscedasticLayer))
    with torch.no_grad():
        # Zeroing the weights makes mu, the low-rank noise and the diagonal scale independent
        # of the input, so the only source of randomness left is the (negligible) diag scale.
        layer.mu_layer.weight.zero_()
        layer.mu_layer.bias.copy_(mu)
        layer.v_layer.weight.zero_()
        layer.v_layer.bias.zero_()
        layer.diag_layer.weight.zero_()
        layer.diag_layer.bias.fill_(-30.0)
    return layer


def test_het_net_layer_takes_num_samples_from_the_method() -> None:
    predictor = _het_net_predictor(num_samples=7)

    layer = next(m for m in predictor.modules() if isinstance(m, HeteroscedasticLayer))

    assert layer.num_samples == 7


def test_het_net_predicts_one_categorical_distribution() -> None:
    predictor = _het_net_predictor()

    distribution = predict(predictor, torch.ones(2, 2))

    assert isinstance(distribution, TorchCategoricalDistribution)
    assert distribution.probabilities.shape == (2, 3)
    torch.testing.assert_close(distribution.probabilities.sum(-1), torch.ones(2))


def test_het_net_representer_calls_the_predictor_once() -> None:
    predictor = _het_net_predictor()
    calls = []
    predictor.register_forward_hook(lambda *_: calls.append(1))

    representation = representer(predictor).represent(torch.ones(2, 2))

    assert len(calls) == 1
    assert isinstance(representation, HetNetRepresentation)
    assert isinstance(representation, TorchCategoricalDistribution)


def test_decompose_dispatches_het_net_representation_to_label_noise_decomposition() -> None:
    representation = _het_net_representation()

    decomposition = decompose(representation)

    assert isinstance(decomposition, LabelNoiseEntropyDecomposition)


def test_quantify_dispatches_het_net_representation_to_label_noise_decomposition() -> None:
    representation = _het_net_representation()

    quantification = quantify(representation)

    assert isinstance(quantification, LabelNoiseEntropyDecomposition)


def test_het_net_decomposition_is_aleatoric_only() -> None:
    representation = _het_net_representation()

    decomposition = decompose(representation)

    aleatoric = decomposition.aleatoric

    assert torch.allclose(decomposition["au"], aleatoric)
    assert aleatoric.shape == (2,)
    with pytest.raises(AttributeError):
        _ = decomposition.total
    with pytest.raises(KeyError):
        _ = decomposition["tu"]
    with pytest.raises(AttributeError):
        _ = decomposition.epistemic
    with pytest.raises(KeyError):
        _ = decomposition["eu"]


def test_measure_het_net_representation_returns_aleatoric_uncertainty() -> None:
    representation = _het_net_representation()

    uncertainty = measure(representation)
    decomposition = decompose(representation)

    assert torch.allclose(uncertainty, decomposition.aleatoric)
    assert uncertainty.shape == (2,)


def test_het_net_aleatoric_is_the_entropy_of_the_prediction() -> None:
    representation = _het_net_representation()

    aleatoric = quantify(representation).aleatoric

    expected = torch.distributions.Categorical(probs=representation.probabilities).entropy()
    torch.testing.assert_close(aleatoric, expected)


@pytest.mark.parametrize("base", [2, "normalize"])
def test_het_net_aleatoric_respects_base(base: float | Literal["normalize"]) -> None:
    representation = _het_net_representation()

    aleatoric = quantify(representation, base=base).aleatoric

    log_base = float(representation.probabilities.shape[-1]) if base == "normalize" else base
    expected = torch.distributions.Categorical(probs=representation.probabilities).entropy() / math.log(log_base)
    torch.testing.assert_close(aleatoric, expected)


def test_het_net_aleatoric_equals_entropy_of_softmax_mu_when_noise_free() -> None:
    """Without label noise every sample is softmax(mu), so the aleatoric uncertainty is its entropy."""
    torch.manual_seed(0)
    predictor = _het_net_predictor(num_samples=50)
    mu = torch.tensor([1.5, -0.5, 0.25])
    _noise_free_layer(predictor, mu)

    aleatoric = quantify(representer(predictor).represent(torch.ones(1, 2))).aleatoric

    expected = torch.distributions.Categorical(probs=torch.softmax(mu, dim=-1)).entropy()
    assert torch.allclose(aleatoric, expected, atol=1e-3)


def test_het_net_aleatoric_is_close_to_log_k_for_very_noisy_zero_mean_input() -> None:
    """Huge zero-mean label noise makes every sample one-hot at a random class, so the average is uniform."""
    torch.manual_seed(0)
    num_classes = 3
    predictor = _het_net_predictor(num_samples=2000)
    layer = _noise_free_layer(predictor, torch.zeros(num_classes))
    with torch.no_grad():
        layer.diag_layer.bias.fill_(50.0)

    aleatoric = quantify(representer(predictor).represent(torch.ones(1, 2))).aleatoric

    assert torch.allclose(aleatoric, torch.tensor(math.log(num_classes)), atol=0.02)
