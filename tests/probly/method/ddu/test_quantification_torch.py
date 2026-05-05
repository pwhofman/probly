"""Tests for torch DDU uncertainty decomposition."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.method.ddu import DDUDensityDecomposition, negative_log_density  # noqa: E402
from probly.method.ddu.torch import TorchDDURepresentation  # noqa: E402
from probly.quantification import measure, quantify  # noqa: E402
from probly.quantification.measure.distribution import entropy  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution  # noqa: E402


def _ddu_representation() -> TorchDDURepresentation:
    return TorchDDURepresentation(
        softmax=TorchProbabilityCategoricalDistribution(torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float64)),
        densities=torch.log(torch.tensor([[0.2, 0.3], [0.1, 0.4]], dtype=torch.float64)),
    )


def test_quantify_dispatches_to_ddu_density_decomposition() -> None:
    decomposition = quantify(_ddu_representation())

    assert isinstance(decomposition, DDUDensityDecomposition)


def test_ddu_density_decomposition_matches_softmax_entropy_and_negative_log_density() -> None:
    representation = _ddu_representation()

    decomposition = quantify(representation)

    assert torch.allclose(decomposition.aleatoric, entropy(representation.softmax), rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        decomposition.epistemic,
        -torch.logsumexp(representation.densities, dim=-1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_negative_log_density_uses_total_gmm_log_density() -> None:
    densities = torch.log(torch.tensor([[0.2, 0.3], [0.1, 0.4]], dtype=torch.float64))

    measured = negative_log_density(densities)

    assert torch.allclose(measured, -torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)))


def test_ddu_density_decomposition_has_no_total_or_canonical_notion() -> None:
    decomposition = quantify(_ddu_representation())

    assert decomposition["au"] is decomposition.aleatoric
    assert decomposition["eu"] is decomposition.epistemic
    with pytest.raises(KeyError):
        _ = decomposition["tu"]
    with pytest.raises(NotImplementedError):
        decomposition.get_canonical()


def test_measure_ddu_representation_requires_explicit_notion() -> None:
    with pytest.raises(NotImplementedError):
        measure(_ddu_representation())
