"""Tests for ordinal-variance / ordinal-entropy uncertainty measures (torch)."""

from __future__ import annotations

import numpy as np
import pytest


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    return torch


class TestTorchOrdinal:
    """Torch implementations of ordinal-variance / ordinal-entropy."""

    def test_variance_on_distribution(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.ordinal import ordinal_variance  # noqa: PLC0415

        # Force registration of the torch dispatch.
        import probly.quantification.measure.ordinal.torch  # noqa: F401, PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        d = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[0.5, 0.5, 0.0]]))
        out = ordinal_variance(d)
        torch.testing.assert_close(out, torch.tensor([0.25]))

    def test_variance_on_raw_tensor_branch(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.ordinal.torch import (  # noqa: PLC0415
            torch_categorical_ordinal_variance,
        )

        out = torch_categorical_ordinal_variance(torch.tensor([[0.5, 0.5]]))
        torch.testing.assert_close(out, torch.tensor([0.25]))

    def test_entropy_on_distribution(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.ordinal import ordinal_entropy  # noqa: PLC0415
        import probly.quantification.measure.ordinal.torch  # noqa: F401, PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        d = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[0.5, 0.5]]))
        out = ordinal_entropy(d)
        # Uniform 2-class -> log(2) in nats.
        torch.testing.assert_close(out, torch.tensor([float(np.log(2))]), atol=1e-5, rtol=1e-5)

    def test_entropy_normalized(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.ordinal import ordinal_entropy  # noqa: PLC0415
        import probly.quantification.measure.ordinal.torch  # noqa: F401, PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        d = TorchProbabilityCategoricalDistribution(tensor=torch.tensor([[0.5, 0.5]]))
        out = ordinal_entropy(d, base="normalize")
        torch.testing.assert_close(out, torch.tensor([1.0]), atol=1e-5, rtol=1e-5)
