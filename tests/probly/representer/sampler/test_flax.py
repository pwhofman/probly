"""Tests for ``probly.representer.sampler.flax``."""

from __future__ import annotations

import pytest


def _flax_nnx():
    pytest.importorskip("flax")
    from flax import nnx  # noqa: PLC0415

    return nnx


class TestSamplerFlax:
    """Flax dropout modules switch to deterministic=False during sampling."""

    def test_flax_dropout_switched_to_train(self) -> None:
        nnx = _flax_nnx()

        from probly.representer.sampler._common import get_sampling_predictor  # noqa: PLC0415
        import probly.representer.sampler.flax  # noqa: F401, PLC0415

        rngs = nnx.Rngs(0)

        class Net(nnx.Module):
            def __init__(self) -> None:
                self.dropout = nnx.Dropout(rate=0.5, deterministic=True, rngs=rngs)

            def __call__(self, x):  # noqa: ANN204
                return self.dropout(x)

        model = Net()
        assert model.dropout.deterministic is True
        sampling_model, cleanup = get_sampling_predictor(model)
        # During sampling, deterministic is flipped to False.
        assert sampling_model.dropout.deterministic is False
        cleanup()
        # After cleanup, the flag is restored.
        assert sampling_model.dropout.deterministic is True
