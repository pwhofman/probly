from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402


@pytest.fixture
def sample_classification_data() -> tuple[Tensor, Tensor]:
    inputs = torch.randn(2, 3, 5, 5)
    targets = torch.randint(0, 2, (2,))
    return inputs, targets


@pytest.fixture
def sample_outputs(
    torch_conv_linear_model: nn.Module,
) -> tuple[Tensor, Tensor]:
    outputs = torch_conv_linear_model(torch.randn(2, 3, 5, 5))
    targets = torch.randint(0, 2, (2,))
    return outputs, targets
