"""Test for torch classification models."""

from __future__ import annotations

import torch
from torch import nn

from probly.method.evidential.classification import (
    EvidentialClassificationDecomposition,
    evidential_classification,
)
from probly.quantification import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty
from probly.quantification.measure.distribution import vacuity
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution
from probly.transformation.dirichlet_clipped_exp_one_activation.torch import _AddOne, _ClippedExp
from tests.probly.torch_utils import count_layers


def test_evidential_classification_appends_clipped_exp_plus_one_on_linear(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    model = evidential_classification(torch_model_small_2d_2d)

    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    count_clipped_exp_original = count_layers(torch_model_small_2d_2d, _ClippedExp)
    count_add_one_original = count_layers(torch_model_small_2d_2d, _AddOne)
    count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

    count_linear_modified = count_layers(model, nn.Linear)
    count_clipped_exp_modified = count_layers(model, _ClippedExp)
    count_add_one_modified = count_layers(model, _AddOne)
    count_sequential_modified = count_layers(model, nn.Sequential)

    assert model is not None
    assert isinstance(model, type(torch_model_small_2d_2d))
    assert count_linear_original == count_linear_modified
    assert count_clipped_exp_original == (count_clipped_exp_modified - 1)
    assert count_add_one_original == (count_add_one_modified - 1)
    assert count_sequential_original == (count_sequential_modified - 1)


def test_evidential_classification_appends_clipped_exp_plus_one_on_conv(
    torch_conv_linear_model: nn.Sequential,
) -> None:
    model = evidential_classification(torch_conv_linear_model)

    count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
    count_clipped_exp_original = count_layers(torch_conv_linear_model, _ClippedExp)
    count_add_one_original = count_layers(torch_conv_linear_model, _AddOne)
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

    count_linear_modified = count_layers(model, nn.Linear)
    count_clipped_exp_modified = count_layers(model, _ClippedExp)
    count_add_one_modified = count_layers(model, _AddOne)
    count_sequential_modified = count_layers(model, nn.Sequential)
    count_conv_modified = count_layers(model, nn.Conv2d)

    assert model is not None
    assert isinstance(model, type(torch_conv_linear_model))
    assert count_linear_original == count_linear_modified
    assert count_clipped_exp_original == (count_clipped_exp_modified - 1)
    assert count_add_one_original == (count_add_one_modified - 1)
    assert count_sequential_original == (count_sequential_modified - 1)
    assert count_conv_original == count_conv_modified


def test_evidential_classification_alpha_is_at_least_one(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """The clipped-exp + 1 parameterization must yield ``alpha >= 1`` everywhere."""
    model = evidential_classification(torch_model_small_2d_2d)
    alpha = model(torch.randn(8, 2))
    assert torch.all(alpha >= 1.0)


# Tests for EvidentialClassificationDecomposition on Torch Dirichlet representations.


def _torch_dirichlet() -> TorchDirichletDistribution:
    alphas = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [10.0, 10.0, 10.0],
            [2.0, 3.0, 5.0],
            [100.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return TorchDirichletDistribution(alphas=alphas)


def test_torch_decomposition_epistemic_matches_vacuity() -> None:
    distribution = _torch_dirichlet()

    decomposition = EvidentialClassificationDecomposition(distribution)

    assert torch.allclose(decomposition.epistemic, vacuity(distribution), rtol=1e-12, atol=1e-12)


def test_torch_decomposition_components_only_epistemic() -> None:
    decomposition = EvidentialClassificationDecomposition(_torch_dirichlet())

    assert decomposition.components == [EpistemicUncertainty]
    assert len(decomposition) == 1


def test_torch_decomposition_canonical_notion_is_epistemic() -> None:
    decomposition = EvidentialClassificationDecomposition(_torch_dirichlet())

    assert decomposition.canonical_notion is EpistemicUncertainty
    assert decomposition.get_canonical() is decomposition.epistemic


def test_torch_decomposition_does_not_expose_aleatoric_or_total() -> None:
    """The paper has no aleatoric / total measures; decomposition must reflect that."""
    decomposition = EvidentialClassificationDecomposition(_torch_dirichlet())

    import pytest  # noqa: PLC0415

    with pytest.raises(KeyError):
        decomposition[AleatoricUncertainty]
    with pytest.raises(KeyError):
        decomposition[TotalUncertainty]
    with pytest.raises(KeyError):
        decomposition["au"]
    with pytest.raises(KeyError):
        decomposition["tu"]


def test_torch_decomposition_caches_component() -> None:
    decomposition = EvidentialClassificationDecomposition(_torch_dirichlet())

    epistemic = decomposition.epistemic

    assert decomposition.epistemic is epistemic
    assert decomposition[EpistemicUncertainty] is epistemic


def test_torch_decomposition_propagates_gradients() -> None:
    alphas = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    distribution = TorchDirichletDistribution(alphas=alphas)

    decomposition = EvidentialClassificationDecomposition(distribution)
    decomposition.epistemic.backward()

    grad = alphas.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_torch_decomposition_on_evidential_model_output(torch_model_small_2d_2d: nn.Sequential) -> None:
    """End-to-end: an EDL-trained classifier's Dirichlet output feeds the decomposition cleanly."""
    model = evidential_classification(torch_model_small_2d_2d)
    alpha = model(torch.randn(5, 2))
    distribution = TorchDirichletDistribution(alphas=alpha)

    decomposition = EvidentialClassificationDecomposition(distribution)

    assert decomposition.epistemic.shape == (5,)
    assert torch.all(decomposition.epistemic > 0.0)
    assert torch.all(decomposition.epistemic <= 1.0)


# Representer dispatch tests.

from probly.method.evidential.classification import (  # noqa: E402
    EvidentialClassificationRepresentation,
    EvidentialClassificationRepresenter,
)
from probly.predictor import predict  # noqa: E402
from probly.quantification import quantify  # noqa: E402
from probly.quantification.decomposition.entropy import SecondOrderEntropyDecomposition  # noqa: E402
from probly.representer import representer  # noqa: E402


def test_representer_factory_returns_edl_representer(torch_model_small_2d_2d: nn.Sequential) -> None:
    model = evidential_classification(torch_model_small_2d_2d)
    rep = representer(model)
    assert isinstance(rep, EvidentialClassificationRepresenter)


def test_representer_output_marked_as_edl_representation(torch_model_small_2d_2d: nn.Sequential) -> None:
    model = evidential_classification(torch_model_small_2d_2d)
    out = representer(model)(torch.randn(5, 2))

    assert isinstance(out, TorchDirichletDistribution)
    assert isinstance(out, EvidentialClassificationRepresentation)


def test_quantify_on_representer_output_dispatches_to_edl_decomposition(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """quantify(representer(model)(x)) -> EvidentialClassificationDecomposition."""
    model = evidential_classification(torch_model_small_2d_2d)
    out = representer(model)(torch.randn(5, 2))

    decomposition = quantify(out)

    assert isinstance(decomposition, EvidentialClassificationDecomposition)


def test_direct_predict_output_not_marked_keeps_default_decomposition(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """quantify(predict(model, x)) -> SecondOrderEntropyDecomposition (UNCHANGED)."""
    model = evidential_classification(torch_model_small_2d_2d)
    direct = predict(model, torch.randn(5, 2))

    assert not isinstance(direct, EvidentialClassificationRepresentation)
    assert isinstance(quantify(direct), SecondOrderEntropyDecomposition)
