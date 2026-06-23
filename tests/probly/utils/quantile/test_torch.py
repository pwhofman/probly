"""Tests for the torch backend of ``calculate_quantile``."""

from __future__ import annotations

import pytest


def _torch():
    return pytest.importorskip("torch")


class TestQuantileTorch:
    """`calculate_quantile` for torch tensors.

    Note: the torch implementation hard-codes ``torch.tensor(1.0)`` for the upper
    cap on the q-level, which is float32. ``torch.quantile`` requires matching
    dtypes between its input and the q tensor, so callers must pass float32 input.
    """

    def test_torch_quantile_runs(self) -> None:
        torch = _torch()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
        q = calculate_quantile(scores, alpha=0.1)
        assert isinstance(q, float)

    def test_torch_alpha_out_of_range_raises(self) -> None:
        torch = _torch()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="alpha must be in"):
            calculate_quantile(torch.tensor([0.1, 0.2], dtype=torch.float32), alpha=1.5)

    def test_torch_empty_scores_raises(self) -> None:
        torch = _torch()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="empty"):
            calculate_quantile(torch.tensor([], dtype=torch.float32), alpha=0.1)

    def test_torch_weighted_quantile_unweighted(self) -> None:
        torch = _torch()
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_weighted_quantile(values, 0.5) == pytest.approx(3.0)

    def test_torch_weighted_quantile_with_weights(self) -> None:
        torch = _torch()
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([1.0, 0.0, 0.0])
        result = calculate_weighted_quantile(values, 0.5, sample_weight=weights)
        assert result == pytest.approx(1.0)
