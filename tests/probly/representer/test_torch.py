"""Tests for representers combined with torch modules."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from probly.representer import Representer


class _TorchModuleRepresenter(torch.nn.Module, Representer):
    def __init__(self, predictor: torch.nn.Linear) -> None:
        torch.nn.Module.__init__(self)
        Representer.__init__(self, predictor)

    def represent(self, inputs: torch.Tensor) -> TorchProbabilityCategoricalDistribution:
        return TorchProbabilityCategoricalDistribution(torch.softmax(self.predictor(inputs), dim=-1))


def test_representer_registers_predictor_as_child_module() -> None:
    predictor = torch.nn.Linear(2, 2)
    representer = _TorchModuleRepresenter(predictor)

    parameters = dict(representer.named_parameters())
    assert parameters["predictor.weight"] is predictor.weight
    assert parameters["predictor.bias"] is predictor.bias
    assert set(representer.state_dict()) == {"predictor.weight", "predictor.bias"}

    representer.eval()
    assert not predictor.training

    representer.to(dtype=torch.float64)
    assert predictor.weight.dtype == torch.float64
